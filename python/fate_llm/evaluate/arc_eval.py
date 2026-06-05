def evaluate_arc_mc_accuracy(model, tokenizer, split="validation", max_examples=None):
    import torch
    from datasets import load_dataset

    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)

    model.eval()
    device = next(model.parameters()).device

    num_examples = 0
    num_correct = 0
    loss_sum = 0.0

    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break

        question = ex["question"]
        choices_text = ex["choices"]["text"]
        choices_label = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        try:
            gold_idx = choices_label.index(answer_key)
        except ValueError:
            continue

        lines = [f"Question: {question}"]
        for lab, txt in zip(choices_label, choices_text):
            lines.append(f"{lab}. {txt}")
        lines.append("Answer:")
        prompt = "\n".join(lines)

        with torch.no_grad():
            choice_losses = []
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = int(prompt_enc["input_ids"].shape[1])

            for choice in choices_text:
                full_text = prompt + " " + choice
                full_enc = tokenizer(full_text, return_tensors="pt")

                input_ids = full_enc["input_ids"].to(device)
                attention_mask = full_enc["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                choice_losses.append(float(outputs.loss.item()))

        num_examples += 1
        loss_sum += choice_losses[gold_idx]

        pred_idx = int(min(range(len(choice_losses)), key=lambda k: choice_losses[k]))
        if pred_idx == gold_idx:
            num_correct += 1

    if num_examples == 0:
        return {
            "arc_mc_accuracy": 0.0,
            "arc_mc_avg_choice_loss": float("inf"),
            "arc_mc_num_examples": 0.0,
        }

    return {
        "arc_mc_accuracy": float(num_correct) / float(num_examples),
        "arc_mc_avg_choice_loss": float(loss_sum) / float(num_examples),
        "arc_mc_num_examples": float(num_examples),
    }
