from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from argparse import ArgumentParser
import tqdm
import torch

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Translator script", 
    )
    parser.add_argument('--src_lang_code', type=str)
    parser.add_argument('--tgt_lang_code', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)

    args, unknown = parser.parse_known_args()

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = args.src_lang_code


    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    with open(args.output_path, 'w') as fo:
        for line in tqdm.tqdm(open(args.input_path).readlines()):
            line = line.strip()
            encoded = tokenizer(line, return_tensors="pt")
            encoded = encoded.to(device)
            generated_tokens = model.generate(**encoded, num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang_code])
            out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            fo.write(out + '\n')
