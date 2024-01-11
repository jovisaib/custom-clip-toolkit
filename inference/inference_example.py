import torch
import clip


def run_doctext_inference(model, preprocess, image, queries, topk):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_types = queries
    image = preprocess(image).unsqueeze(0).to(device)
    text = torch.cat([clip.tokenize(c) for c in text_types]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(topk)

    predicted_class = text_types[indices[0]]

    if predicted_class == text_types[0]:
        predicted_class = 'machine_text'
    else:
        predicted_class = 'handwritten_text'

    return predicted_class