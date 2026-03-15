from ingest import build_index
from retriever import retrieve
from core.llm import LegalLLM
import os


def main():

    print("\n==============================")
    print(" LegalMind Contract Analyzer ")
    print("==============================\n")

    # check if contract file exists
    file_path = "data/sample_contract.txt"

    if not os.path.exists(file_path):
        print("Error: Contract file not found.")
        print("Expected location:", file_path)
        return

    print("Building document index...\n")

    # build vector index
    embedder, store = build_index(file_path)

    print("Index built successfully.")

    print("\nLoading Legal LLM...\n")

    # load local model
    llm = LegalLLM()

    print("\nSystem ready. Ask your questions.")

    while True:

        query = input("\nAsk a contract question (type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            print("\nExiting LegalMind. Goodbye!\n")
            break

        if query == "":
            print("Please enter a valid question.")
            continue

        try:
            # retrieve relevant clauses
            chunks = retrieve(query, embedder, store)

            if not chunks:
                print("No relevant clauses found.")
                continue

            # combine retrieved context
            context = "\n".join(chunks)

            print("\nAnalyzing contract...\n")

            # generate response
            response = llm.generate(context, query)

            print("\n--- Analysis Result ---\n")
            print(response)

        except Exception as e:
            print("Error during analysis:", str(e))


if __name__ == "__main__":
    main()