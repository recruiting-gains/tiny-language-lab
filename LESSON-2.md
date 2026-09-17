# Lesson 2 — a prediction, a mistake, an update

Our first trainable model is a **bigram baseline**. It uses the current character to predict the next one. Attention and longer context come later.

## What actually happened

The model began with random scores, not downloaded knowledge. There are seven characters in our tiny vocabulary, so it has 7 × 7 = 49 learnable scores.

Training repeats three actions:

1. Predict a probability for each possible next character.
2. Compare those probabilities with the correct next character.
3. Adjust the scores to reduce the error.

**Loss** is the numerical measure of that error. When the correct next character is `a`, assigning `a` a very low probability produces a bigger loss than assigning it a high probability. Lower loss is better for the examples being evaluated; it is not the same thing as accuracy or universal ability.

## Results from the real first run

| Measurement | Before | After 200 training steps |
|---|---:|---:|
| Probability of `a` following `c` | 21.3% | 99.0% |
| Training cross-entropy loss | 2.2156 | 0.1511 |
| Held-out toy-pattern cross-entropy loss | 2.2022 | 0.1609 |

The held-out strings were not used for weight updates, but share the same very simple cat/dog patterns. Improvement is evidence of learning this toy pattern—not understanding animals, general language, or the meaning of words.

Actual generation before training:

```text
cgtdaogaaaotoda cgc tdd oottcgo gaocgoot 
```

Actual generation after training:

```text
cat dog cat dog cg dt dog dog cat dog cat
```

The mistakes are intentionally left in. We sampled from the model's probabilities; we did not replace its output with a scripted correct answer.

The checkpoint was saved and reloaded; the prediction scores matched exactly. Sixteen tests passed across the tokenizer and baseline at this stage. The subsequent transformer, complete training resume, and browser delivery are now implemented; see README.md for the current project.
