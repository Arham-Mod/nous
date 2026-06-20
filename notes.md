## Day 1

### Wrote allocate_ranks()
- Test 1 passed: uniform sensitivity → uniform rank 8
- Test 2 showed: v_proj layers getting higher rank than q_proj
- Budget check: adaptive total close to uniform total 512
- Small difference due to clamping and rounding — acceptable

### Next
- learn about the layers of transformers and what q and v pass forward as one should know why a particular layer has more senstivity than other(refer blackblue or whatever it is)
- Run Notebook 02 on Colab to get real sensitivity scores
- Run allocate_ranks() on real scores

## Day 2
## allocate_ranks() — final fix

Bug: upgrade step sorted by sensitivity, fully upgrading 
floor→ceiling for top sensitivity layers regardless of how 
close their raw value was to ceiling. Drained budget before 
moderate layers could land in between, causing rank 8 to vanish.

Fix: sort by closeness-to-ceiling instead of raw sensitivity.
Layers nearest their ceiling get upgraded first. Now distribution
shows a proper spread: 2, 4, 8, 16 all represented.

Final test results:
- Uniform sensitivity → all rank 8 (correct)
- Varied sensitivity → spread across 2/4/8/16, v_proj dominates top
- Budget: 512 = 512 exact match