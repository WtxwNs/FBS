"""Held-out perplexity and multiple-choice evaluation from prepared local records.

Records contain prompt_ids and either answer_ids, or choice_ids plus gold_index.
Task templates and few-shot exemplars must be supplied explicitly in the input;
the evaluator never borrows calibration prompts or fills missing benchmark data.
"""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .checkpoint import load_checkpoint
from .inference import prefill


@torch.no_grad()
def score(model, prompt, continuation, threshold):
    if not prompt or not continuation:
        raise ValueError("Prompt and scored continuation must be nonempty")
    device = next(model.parameters()).device
    inputs = torch.tensor([prompt + continuation[:-1]], device=device)
    logits, state = prefill(model, inputs, threshold)
    targets = torch.tensor(continuation, device=device)
    nll = F.cross_entropy(logits[0, len(prompt)-1:].float(), targets, reduction="sum").item()
    return nll, len(continuation), state


def evaluate(model, rows, threshold=.8):
    predictions = []
    total_nll, total_tokens, correct, classified = 0., 0, 0, 0
    for index, row in enumerate(rows):
        if row.get("split") in ("train", "dev", "calibration"):
            raise ValueError("Evaluation input contains training/development records")
        result = {"id": row.get("id", index), "threshold": threshold}
        if "choice_ids" in row:
            values = [score(model, row["prompt_ids"], option, threshold)[0] for option in row["choice_ids"]]
            prediction = min(range(len(values)), key=values.__getitem__)
            gold = row["gold_index"]
            if not 0 <= gold < len(values):
                raise ValueError("Invalid gold option")
            correct += prediction == gold
            classified += 1
            result.update(choice_nll=values, prediction=prediction, gold=gold, correct=prediction==gold)
        elif "answer_ids" in row:
            nll, count, state = score(model, row["prompt_ids"], row["answer_ids"], threshold)
            total_nll += nll
            total_tokens += count
            result.update(nll=nll, tokens=count, executed=state.executed, skipped=state.skipped)
        else:
            raise ValueError("Each record needs choice_ids or answer_ids")
        predictions.append(result)
    mean_nll = total_nll/total_tokens if total_tokens else None
    summary = dict(records=len(predictions), classified=classified,
                   accuracy=correct/classified if classified else None,
                   scored_tokens=total_tokens, mean_nll=mean_nll,
                   perplexity=math.exp(mean_nll) if mean_nll is not None and mean_nll < 700 else None)
    return predictions, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--thresholds',nargs='+',type=float,default=[1.,.9,.8,.7])
    parser.add_argument('--shots',type=int,default=5)
    args=parser.parse_args()
    rows=[json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not rows or any(row.get('shots') != args.shots for row in rows):
        raise ValueError('Input must record the actual shot count for every prompt')
    model,_=load_checkpoint(args.checkpoint,args.device);model.eval()
    destination=Path(args.output_dir);destination.mkdir(parents=True,exist_ok=True)
    report=[]
    for threshold in args.thresholds:
        predictions,summary=evaluate(model,rows,threshold)
        summary['threshold']=threshold
        report.append(summary)
        (destination/f'predictions_{threshold:g}.jsonl').write_text('\n'.join(json.dumps(row) for row in predictions)+'\n')
    (destination/'summary.json').write_text(json.dumps({'protocol':vars(args),'parameters':model.parameter_report(),'results':report},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
