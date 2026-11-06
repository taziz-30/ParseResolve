from flask import Flask, render_template, request
from nltk import CFG, ChartParser, Tree, PCFG, ViterbiParser
from nltk.grammar import ProbabilisticProduction
from collections import Counter
from decimal import Decimal
import re
import benepar
app = Flask(__name__)
neural_parser = None
neural_parser_error = None
def get_neural_parser():
    global neural_parser, neural_parser_error
    if neural_parser is None and neural_parser_error is None:
        try:
            neural_parser = benepar.Parser("benepar_en3")
        except Exception as error:
            import traceback
            neural_parser_error = str(error)
            print("NEURAL PARSER ERROR:", flush=True)
            traceback.print_exc()
    return neural_parser
grammar = CFG.fromstring("""
S -> NP VP
NP -> PRON | DET NOUN | DET ADJ NOUN | DET NOUN PP | DET ADJ NOUN PP | NAME
VP -> VERB | VERB NP | VERB NP PP | VERB PP | ADV VERB NP | VERB ADV NP
PP -> PREP NP
PRON -> 'i' | 'you' | 'he' | 'she' | 'we' | 'they'
DET -> 'the' | 'a' | 'an' | 'my' | 'his' | 'her' | 'their'
ADJ -> 'big' | 'small' | 'young' | 'old' | 'red' | 'beautiful'
ADV -> 'quickly' | 'slowly'
NOUN -> 'man' | 'woman' | 'boy' | 'girl' | 'dog' | 'cat' | 'bird'
NOUN -> 'telescope' | 'plane' | 'book' | 'car' | 'student' | 'teacher'
NOUN -> 'school' | 'apple' | 'ball' | 'room' | 'park' | 'robot'
NOUN -> 'computer' | 'phone' | 'friend' | 'food'
VERB -> 'saw' | 'see' | 'sees' | 'reads' | 'read'
VERB -> 'hit' | 'hits' | 'likes' | 'like' | 'loves' | 'love'
VERB -> 'helped' | 'help' | 'helps' | 'found' | 'find' | 'finds'
VERB -> 'chased' | 'chases' | 'chase' | 'watched' | 'watch' | 'watches'
VERB -> 'carried' | 'carry' | 'carries' | 'ate' | 'eats' | 'eat'
VERB -> 'met' | 'meet' | 'meets' | 'opened' | 'open' | 'opens'
VERB -> 'is' | 'are' | 'was' | 'were'
PREP -> 'with' | 'near' | 'in' | 'on' | 'under' | 'beside'
PREP -> 'behind' | 'at' | 'by' | 'of'
NAME -> 'john' | 'mary' | 'alice' | 'bob'
""")
parser = ChartParser(grammar)
productions = grammar.productions()
rule_counts = Counter(p.lhs() for p in productions)
probabilistic_productions = [
    ProbabilisticProduction(p.lhs(), p.rhs(),
                           prob=1.0 / rule_counts[p.lhs()])
    for p in productions
]
probabilistic_grammar = PCFG(grammar.start(), probabilistic_productions)
probabilistic_parser = ViterbiParser(probabilistic_grammar)
def tokenize(sentence):
    sentence = re.sub(r"[.!?]+$", "", sentence.lower().strip())
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence)
def parse_sentence(sentence):
    tokens = tokenize(sentence)
    if not tokens or len(tokens) > 25:
        return []
    try:
        return list(parser.parse(tokens))[:10]
    except (ValueError, TypeError):
        return []
def probabilistic_parse(sentence):
    tokens = tokenize(sentence)
    if not tokens or len(tokens) > 25:
        return None
    try:
        return next(probabilistic_parser.parse(tokens), None)
    except (ValueError, TypeError):
        return None
def neural_parse(sentence):
    model = get_neural_parser()
    if model is None:
        return None, neural_parser_error or "Neural model could not be loaded."
    try:
        return model.parse(sentence), None
    except Exception as error:
        return None, str(error)
def find_child(tree, label):
    if tree is None:
        return None
    return next((child for child in tree
                 if isinstance(child, Tree) and child.label() == label), None)
def words_of(tree):
    return tree.leaves() if tree is not None else []
def phrase_text(tree):
    words = words_of(tree)
    return "the speaker" if words == ["i"] else " ".join(words)
def get_meaning(tree):
    subject, vp = find_child(tree, "NP"), find_child(tree, "VP")
    if subject is None or vp is None:
        return "A grammatical structure was found."
    subject_text = phrase_text(subject).capitalize()
    verb_node, object_node = find_child(vp, "VERB"), find_child(vp, "NP")
    verb_text, object_text = " ".join(words_of(verb_node)), phrase_text(object_node)
    vp_pp = find_child(vp, "PP")
    if vp_pp is not None:
        pp_text = " ".join(words_of(vp_pp))
        if verb_text == "saw" and object_text == "the man" and pp_text == "with a telescope":
            return ("The speaker used a telescope to see the man. "
                    "The phrase 'with a telescope' describes the action of seeing.")
        return (f"{subject_text} {verb_text} {object_text} in relation to '{pp_text}'. "
                f"The phrase '{pp_text}' describes the action.")
    object_pp = find_child(object_node, "PP")
    if object_pp is not None:
        pp_text = " ".join(words_of(object_pp))
        object_words, pp_words = words_of(object_node), words_of(object_pp)
        base_object = " ".join(object_words[:-len(pp_words)])
        if verb_text == "saw" and base_object == "the man" and pp_text == "with a telescope":
            return ("The speaker saw a man who had a telescope. "
                    "The phrase 'with a telescope' describes the man.")
        return f"{subject_text} {verb_text} {base_object}, and '{pp_text}' describes that object."
    if verb_text and object_text:
        return f"{subject_text} {verb_text} {object_text}."
    if verb_text:
        return f"{subject_text} {verb_text}."
    return "A grammatical structure was found."
@app.route("/", methods=["GET", "POST"])
def home():
    result = {
        "sentence": "", "trees": [], "message": "", "status": "empty",
        "ambiguity_status": "", "probabilistic_tree": "", "probability": "",
        "neural_tree": "", "neural_error": ""
    }
    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        result["sentence"] = sentence
        if not sentence:
            result.update(message="Please enter a sentence.", status="error")
        elif len(sentence) > 500:
            result.update(message="Please enter a sentence under 500 characters.",
                          status="error")
        elif not tokenize(sentence):
            result.update(message="Invalid input. Please enter a sentence containing English words.",
                          status="error")
        else:
            parsed_trees = parse_sentence(sentence)
            if parsed_trees:
                count = len(parsed_trees)
                result["ambiguity_status"] = (
                    f"Ambiguous Sentence Detected — {count} possible interpretations found."
                    if count > 1 else
                    "One Parse Found — No ambiguity detected by the current grammar."
                )
                for number, tree in enumerate(parsed_trees, 1):
                    vp = find_child(tree, "VP")
                    obj = find_child(vp, "NP") if vp else None
                    if vp and find_child(vp, "PP"):
                        attachment = "VP Attachment"
                    elif obj and find_child(obj, "PP"):
                        attachment = "NP Attachment"
                    else:
                        attachment = ""
                    result["trees"].append({
                        "number": number,
                        "text": tree.pformat(margin=80),
                        "meaning": get_meaning(tree),
                        "attachment": attachment
                    })
                best_tree = probabilistic_parse(sentence)
                if best_tree is not None:
                    result["probabilistic_tree"] = best_tree.pformat(margin=80)
                    result["probability"] = format(Decimal(str(best_tree.prob())), "f")
            neural_tree, result["neural_error"] = neural_parse(sentence)
            if neural_tree is not None:
                result["neural_tree"] = neural_tree.pformat(margin=80)
            successful = [
                name for name, found in (
                    ("Rule-Based", bool(result["trees"])),
                    ("Probabilistic", bool(result["probabilistic_tree"])),
                    ("Neural", bool(result["neural_tree"]))
                ) if found
            ]
            if successful:
                result["status"] = "success"
                result["message"] = (
                    "Analysis completed successfully. Working models: "
                    + ", ".join(successful) + "."
                )
                if not result["trees"] and result["neural_tree"]:
                    result["message"] += (
                        " The sentence was not recognized by the current rule-based "
                        "grammar, but Neural Parsing generated a parse tree."
                    )
            else:
                result.update(
                    status="error",
                    message=("No model could generate a parse tree for this input. "
                             "Please check the sentence and try again.")
                )
    return render_template("index.html", **result)
if __name__ == "__main__":
    app.run(debug=True)
