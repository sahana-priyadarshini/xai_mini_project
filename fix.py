content = open('03_explain.py', 'r', encoding='utf-8').read()

old = """    pg_explainer.algorithm.train(
        epoch=1,
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        target=out[all_labelled].argmax(dim=-1),
        index=torch.tensor(all_labelled),
        edge_type=data.edge_type,
    )
    print("[PGExplainer] Training done.")"""

new = """    for epoch in range(1, 31):
        for nidx in all_labelled[:200]:
            pg_explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=data.x,
                edge_index=data.edge_index,
                target=out[nidx].argmax(dim=-1).unsqueeze(0),
                index=nidx,
                edge_type=data.edge_type,
            )
    print("[PGExplainer] Training done.")"""

if old in content:
    content = content.replace(old, new)
    open('03_explain.py', 'w', encoding='utf-8').write(content)
    print('Fixed!')
else:
    print('Pattern not found - already fixed or different content')