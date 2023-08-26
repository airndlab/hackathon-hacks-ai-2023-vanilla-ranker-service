import uvicorn
from fastapi import FastAPI

import model_util

app = FastAPI(
    title='Vanilla-Ranker API'
)


@app.get("/find_similarity")
async def find_similarity(question: str, vectorizer: str = None):
    answer = model_util.get_answer(question, vectorizer)
    return [{
        'q': question,
        'a': answer,
        'type': 'question',
        # TODO: id?
        'id': 'None',
        # TODO: weight?
        'weight': 0.90
    }]


@app.get("/train")
async def train():
    # TODO: Оно тут нжуно?
    return 'Not implemented'


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(8086))
