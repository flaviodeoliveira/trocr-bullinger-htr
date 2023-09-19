import gradio as gr
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor
# import utils
import base64
# from datasets import load_metric
import evaluate
import logging

# Only show log messages that are at the ERROR level or above, effectively filtering out any warnings
logging.getLogger('transformers').setLevel(logging.ERROR)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
image_processor = AutoImageProcessor.from_pretrained("pstroe/bullinger-general-model")
model = VisionEncoderDecoderModel.from_pretrained("pstroe/bullinger-general-model")

# Create examples
# Get images and respective transcriptions from the examples directory
def get_example_data(folder_path="./examples/"):
    
    example_data = []
    
    # Get list of all files in the folder
    all_files = os.listdir(folder_path)
    
    # Loop through the file list
    for file_name in all_files:
        
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is an image (.png)
        if file_name.endswith(".png"):
            
            # Construct the corresponding .txt filename (same name)
            corresponding_text_file_name = file_name.replace(".png", ".txt")
            corresponding_text_file_path = os.path.join(folder_path, corresponding_text_file_name)
            
            # Initialize to a default value
            transcription = "Transcription not found."
            
            # Try to read the content from the .txt file
            try:
                with open(corresponding_text_file_path, "r") as f:
                    transcription = f.read().strip()
            except FileNotFoundError:
                pass  # If the corresponding .txt file is not found, leave the default value
            
            example_data.append([file_path, transcription])
            
    return example_data

# From pstroe's script
# def compute_metrics(pred):

#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
#     label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

#     cer = cer_metric.compute(predictions=pred_str, references=label_str)

#     return {"cer": cer}

def process_image(image, ground_truth):

    cer = None

    # prepare image
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # generate (no beam search)
    generated_ids = model.generate(pixel_values)

    # decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if ground_truth is not None and ground_truth.strip() != "":

        # Debug: Print lengths before computing metric
        print("Number of predictions:", len(generated_text))
        print("Number of references:", len(ground_truth))

        # Check if lengths match
        if len(generated_text) != len(ground_truth):

            print("Mismatch in number of predictions and references.")
            print("Predictions:", generated_text)
            print("References:", ground_truth)
            print("\n")

        cer = cer_metric.compute(predictions=[generated_text], references=[ground_truth])
        # cer = f"{cer:.3f}"

    else:

        cer = "Ground truth not provided"
    
    return generated_text, cer

# One way to use .svg files
# logo_url = "https://www.bullinger-digital.ch/bullinger-digital.svg"
# logo_url = "https://www.cl.uzh.ch/docroot/logos/uzh_logo_e_pos.svg"

# header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='180px'>".format(
#     utils.img_to_bytes(".uzh_logo_e_pos.svg")
# )

# Encode images
with open("assets/uzh_logo_mod.png", "rb") as img_file:
    logo_html = base64.b64encode(img_file.read()).decode('utf-8')

# with open("assets/bullinger-digital.png", "rb") as img_file:
with open("assets/bullinger_logo.png", "rb") as img_file:
    footer_html = base64.b64encode(img_file.read()).decode('utf-8')

# App header
title = """
    <h1 style='text-align: center'> TrOCR: Bullinger Dataset</p>
"""

description = """
    Use of Microsoft's [TrOCR](https://arxiv.org/abs/2109.10282), an encoder-decoder model consisting of an \
    image Transformer encoder and a text Transformer decoder for state-of-the-art optical character recognition \
    (OCR) and handwritten text recognition (HTR) on text line images. \
    This particular model was fine-tuned on [Bullinger Dataset](https://github.com/pstroe/bullinger-htr) \
    as part of the project [Bullinger Digital](https://www.bullinger-digital.ch)
    ([References](https://www.cl.uzh.ch/de/people/team/compling/pstroebel.html#Publications)).
    * HF `model card`: [pstroe/bullinger-general-model](https://huggingface.co/pstroe/bullinger-general-model) | \
    [Flexible Techniques for Automatic Text Recognition of Historical Documents](https://doi.org/10.5167/uzh-234886)
"""

# articles = """
#     <p style='text-align: center'><a href='https://arxiv.org/abs/2109.10282'>TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models</a><br>
#     <a href='https://doi.org/10.5167/uzh-234886'>Flexible Techniques for Automatic Text Recognition of Historical Documents</a><br>
#     <a href='https://zenodo.org/record/7715357'>Bullingers Briefwechsel zugänglich machen: Stand der Handschriftenerkennung</a></p>
# """

# Read .png and the respective .txt files
examples = get_example_data()

# load_metric() is deprecated
# cer_metric = load_metric("cer")
# pip install jiwer
# pip install evaluate
cer_metric = evaluate.load("cer")

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="TrOCR Bullinger",
) as demo:

    gr.HTML(
        f"""
        <div style='display: flex; justify-content: right; width: 100%;'>
            <img src='data:image/png;base64,{logo_html}' class='img-fluid' width='200px'>
        </div>
        """
    )

    #174x60

    title = gr.HTML(title)
    description = gr.Markdown(description)

    with gr.Row():

        with gr.Column(variant="panel"):

            input = gr.components.Image(type="pil", label="Input image:")

            with gr.Row():

                btn_clear = gr.Button(value="Clear")
                button = gr.Button(value="Submit")

        with gr.Column(variant="panel"):

            output = gr.components.Textbox(label="Generated text:")
            ground_truth = gr.components.Textbox(value="", placeholder="Provide the ground truth, if available.", label="Ground truth:")
            cer_output = gr.components.Textbox(label="CER:")

    with gr.Row():

        with gr.Accordion(label="Choose an example from test set:", open=False):
            
            gr.Examples(
                examples=examples,
                inputs = [input, ground_truth],
                label=None,
            )

    with gr.Row():

        # gr.HTML(
        #     f"""
        #     <div style="display: flex; align-items: center; justify-content: center">
        #         <img src="data:image/png;base64,{footer_html}" style="width: 150px; height: 60px; object-fit: contain; margin-right: 5px; margin-bottom: 5px">
        #         <p style="font-size: 13px">
        #             Bullinger Digital | Institut für Computerlinguistik, Universität Zürich, 2023
        #         </p>
        #     </div>
        #     """
        # )
        gr.HTML(
            f"""
            <div style="display: flex; align-items: center; justify-content: center">
                <img src="data:image/png;base64,{footer_html}" style="height: 40px; object-fit: contain; margin-right: 5px; margin-bottom: 5px">
                <p style="font-size: 13px">
                    <strong>Bullinger</strong><u>Digital</u> | Institut für Computerlinguistik, Universität Zürich, 2023
                </p>
            </div>
            """
        )

    #383x85

    button.click(process_image, inputs=[input, ground_truth], outputs=[output, cer_output])
    btn_clear.click(lambda: [None, "", "", ""], outputs=[input, output, ground_truth, cer_output])

    # # Try to force light mode
    # js = """
    #     function () {
    #         gradioURL = window.location.href
    #         if (!gradioURL.endsWith('?__theme=light')) {
    #             window.location.replace(gradioURL + '?__theme=light');
    #     }
    # }"""

    # demo.load(_js=js)

if __name__ == "__main__":

    demo.launch(favicon_path="icon.png")
