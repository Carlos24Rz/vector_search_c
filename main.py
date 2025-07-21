from pathlib import Path
import sys
import argparse
import tree_sitter_c
from tree_sitter import Language, Parser
from sentence_transformers.util import semantic_search
import torch
import torch.cuda
import torch.mps
from unixcoder import UniXcoder

C_LANGUAGE = Language(tree_sitter_c.language())

ts_parser = Parser(C_LANGUAGE)

ts_query = C_LANGUAGE.query(
    """
(function_definition) @function
""")

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def get_file_functions(file: Path):

    with file.open("rb") as f:
        content = f.read()

    tree = ts_parser.parse(content, encoding="utf8")

    query_captures = ts_query.captures(tree.root_node)

    return [function.text.decode(encoding="utf-8") for function in query_captures.get("function", [])]

def get_project_functions(project_root_dir):
    project_functions = []

    # Find c files in project for parsing function definitions
    for project_file in project_root_dir.glob("**/*.c"):
        project_file_functions = get_file_functions(project_file)

        project_functions.extend(project_file_functions)

    print(f"Found {len(project_functions)} functions in project directory {str(project_root_dir)}")

    return project_functions

def get_project_embeddings(project_functions, model):

    print("Generating embeddings...")

    function_tokens = model.tokenize(project_functions, max_length=512, mode="<encoder-only>", padding=True)

    function_tensors = torch.tensor(function_tokens).to(device)

    token_embeddings, function_embeddings = model(function_tensors)

    return function_embeddings


def query_embeddings(embeddings, query, top_k, model):
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(top_k, len(embeddings))
    
    query_token =  model.tokenize([query], max_length=512, mode="<encoder-only>", padding=True)

    query_tensor = torch.tensor(query_token).to(device) 

    query_token_embeddings, query_embedding = model(query_tensor)

    # Perform semantic search
    similarity_scores = semantic_search(query_embeddings=query_embedding, corpus_embeddings=embeddings, top_k=top_k)

    return similarity_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("project_root", help="Project directory", default=Path(__file__).parent)

    parser.add_argument("--top_k", help="Return top k results for query", type=int, default=5)

    parser.add_argument("--n_limit", help="Limit the list of functions to query to the first n elements (-1 for unlimited)", type=int, default=-1)

    args = parser.parse_args()

    project_root_dir = Path(args.project_root).resolve()

    if not project_root_dir.is_dir():
        print(f"Cannot find project root directory {str(project_root_dir)}", file=sys.stderr)
        sys.exit(1)

    # Get Project embeddings

    project_functions = get_project_functions(project_root_dir)

    # Initialize model
    model =  UniXcoder("microsoft/unixcoder-base-nine")
    model.to(device)

    if args.n_limit >= 0:
        project_functions = project_functions[:args.n_limit]

    project_embeddings = get_project_embeddings(project_functions, model)

    if (project_embeddings.shape[0] == 0):
        print(f"No embeddings were generated for project {str(project_root_dir)}")
        sys.exit(1)

    try:
        while True:
            query = input("Search Query: ").strip()

            if len(query) == 0:
                continue

            query_results = query_embeddings(project_embeddings, query, args.top_k, model)

            print("\nQuery:", query)
            print(f"Top {args.top_k} most similar functions in project:")

            for query_entry in query_results:
                for entry in query_entry:
                    print(f"(Score: {entry['score']:.4f})\n", project_functions[entry["corpus_id"]])
                    print("============================================")
    except KeyboardInterrupt:
        print("\nbye")
