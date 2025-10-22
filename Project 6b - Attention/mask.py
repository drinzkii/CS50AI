import sys
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM


MODEL = "bert-base-uncased"
K = 3  # number of predictions
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200



def main():
    text = input("Text: ")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")

    # Locate [MASK] token index
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Run model and obtain attentions
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Show top-K predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)



def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index (0-based) of the mask token in the tokenized input.
    If not found, return None.
    """
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return None

    try:
        ids = input_ids.numpy()
    except Exception:
        ids = input_ids

    # Handle batched input (batch_size, seq_len)
    if hasattr(ids, "__len__") and len(ids) > 0 and hasattr(ids[0], "__len__"):
        seq = list(ids[0])
    else:
        seq = list(ids)

    for idx, token_id in enumerate(seq):
        if int(token_id) == int(mask_token_id):
            return idx
    return None


def get_color_for_attention_score(attention_score):
    """
    Convert an attention score (0–1) into a grayscale RGB color.
    0 -> black (0,0,0)
    1 -> white (255,255,255)
    """
    s = float(attention_score)
    s = max(0.0, min(1.0, s))  # clamp between 0 and 1
    val = int(round(s * 255))
    return (val, val, val)


def visualize_attentions(tokens, attentions):
    """
    Generate an attention diagram for each layer and head.
    """
    # attentions: tuple of tensors [num_layers][batch, num_heads, seq_len, seq_len]
    for layer_idx, layer_att in enumerate(attentions):
        try:
            num_heads = int(layer_att.shape[1])
        except Exception:
            num_heads = len(layer_att[0])

        for head_idx in range(num_heads):
            att = layer_att[0][head_idx]
            try:
                att_weights = att.numpy()
            except Exception:
                att_weights = att
            # 1-indexed layer/head for filenames
            generate_diagram(layer_idx + 1, head_idx + 1, tokens, att_weights)



def generate_diagram(layer_number, head_number, tokens, attention_scores):
    """
    Generate and save an image visualizing the attention matrix for one head.
    """
    seq_len = len(tokens)
    width = PIXELS_PER_WORD * seq_len
    height = PIXELS_PER_WORD * seq_len

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw attention grid
    for i in range(seq_len):
        for j in range(seq_len):
            score = attention_scores[i][j]
            color = get_color_for_attention_score(score)
            x0 = j * PIXELS_PER_WORD
            y0 = i * PIXELS_PER_WORD
            x1 = x0 + PIXELS_PER_WORD
            y1 = y0 + PIXELS_PER_WORD
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Draw token labels
    for idx, token in enumerate(tokens):
        x = idx * PIXELS_PER_WORD + GRID_SIZE
        y = idx * PIXELS_PER_WORD + GRID_SIZE
        draw.text((x, 5), token, fill=(0, 0, 0), font=FONT)
        draw.text((5, y), token, fill=(0, 0, 0), font=FONT)

    # Save image
    filename = f"Attention_Layer{layer_number}_Head{head_number}.png"
    img.save(filename)
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
