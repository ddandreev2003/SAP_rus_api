from llm_interface.llm_SAP import LLM_SAP

def main():
    description = input("Enter your description for SAP prompt generation: ")
    sap_result = LLM_SAP([description], llm='Qwen')[0]
    print("\nSAP prompt decomposition:")
    print("Explanation:", sap_result.get('explanation', ''))
    print("Prompts list:")
    for i, p in enumerate(sap_result.get('prompts_list', [])):
        print(f"  {i+1}. {p}")
    print("Switch prompt steps:", sap_result.get('switch_prompts_steps', []))

if __name__ == "__main__":
    main()
