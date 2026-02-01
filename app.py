from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn


print("Loading models...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Models loaded!")

app = Flask(__name__)


def get_real_candidates(answer):
    candidates = set()
    answer_lower = answer.lower()

    for syn in wn.synsets(answer):
        for hyper in syn.hypernyms():
            for hypo in hyper.hyponyms():
                for lemma in hypo.lemmas():
                    word = lemma.name().replace("_", " ")
                    if answer_lower not in word.lower():
                        candidates.add(word)

    return list(candidates)


def generate_distractors(question, correct_answer):
    candidates = get_real_candidates(correct_answer)

    # Safety fallback (same as your code)
    if len(candidates) < 3:
        return ["liver", "pancreas", "spleen"]


    answer_emb = model.encode(correct_answer, convert_to_tensor=True)
    cand_embs = model.encode(candidates, convert_to_tensor=True)

    scores = util.cos_sim(answer_emb, cand_embs)[0]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [word for word, _ in ranked[:3]]


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    question = data.get("question", "")
    answer = data.get("answer", "")

    distractors = generate_distractors(question, answer)

    return jsonify({
        "distractors": distractors
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
