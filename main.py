import argparse

def load_config(config_path, args):
    import yaml
    from types import SimpleNamespace
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    f.close()
    
    # updata key
    for key, value in vars(args).items():
        if key not in config: 
            config[key] = value
        
    config = SimpleNamespace(**config)
    
    return config

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Path to config')
    parser.add_argument('--config_path', type=str,default="./config/config.yaml")
    parser.add_argument('--folder_name', type=str,default="0")
    parser.add_argument('--step', type=int,default=0, help=
                        '0: get_propositions_by_llm, '
                        '1: extract_entity_from_propositions'
                        '2: index'
                        '3: retrieve'
                        '4: generation')
    args = parser.parse_args()
    
    config = load_config(config_path=args.config_path, args=args)
    
    if args.step == 0:
        from utils import get_propositions_by_llm, get_propositions_by_split
        get_propositions_by_llm(config)
        
    elif args.step == 1:
        from utils import extract_entity_from_propositions
        extract_entity_from_propositions(config)
        
    elif args.step == 2:
        from utils import index
        index(config)
        
    elif args.step == 3:
        if config.mode == "hypergraph":
            from utils import hyper_graph
            hyper_graph(config)
        elif config.mode == "vallina":
            from utils import vallina_retrieve
            vallina_retrieve(config)
        else:
            pass
            
    elif args.step == 4:
        from utils import generation
        generation(config)
        
    elif args.step == 5:
        from utils import test
        test(config)

    elif args.step == 6:
        from utils import extract_entity_from_question
        extract_entity_from_question(config)
        
    elif args.step == 7:
        from utils import extract_propositions_entity_from_chunks
        extract_propositions_entity_from_chunks(config)
        