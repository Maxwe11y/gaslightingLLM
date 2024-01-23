import lmppl

scorer = lmppl.LM('gpt2')
text = [
    'I dropped my laptop on my knee, and someone stole my coffee.',
    'page hide script head number1855',
    'A bad day good luck at this moment',
    '!!!!!!!!!!!!!!!!!!!!!!!!!!',
    ";) Page! <=`) hide({\</_{\nofollowtext Ha ! Indnofollow <%$}; script{[includegraphics",
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))

print(f"prediction: {text[ppl.index(min(ppl))]}")