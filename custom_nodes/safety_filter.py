from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SafetyFilter:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "safety_filter": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "nsfw"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "int_field": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "print_to_screen": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Safety"

    def get_model_info(self, model_ID, device):
        model = CLIPModel.from_pretrained(model_ID).to(device)
        processor = CLIPProcessor.from_pretrained(model_ID)
        tokenizer = CLIPTokenizer.from_pretrained(model_ID)
        return model, processor, tokenizer

    def test(self, image, safety_filter, int_field, threshold, print_to_screen):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_IDs = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
        model_ID = model_IDs[1]
        model_clip, processor, tokenizer = self.get_model_info(model_ID, device)

        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"


        inputs = processor(text=safety_filter, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model_clip(**inputs)

        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        print(probs)
        print(logits_per_image)

        processed_images = []
        for i, logits_per_image1 in enumerate(logits_per_image):
        # Do some processing on the image, in this example I just invert it
            if (logits_per_image1 / 100.0 > threshold):
                image[i] = 0.0 * image[i]

            processed_images.append(image[i])
        return (processed_images, i)
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {safety_filter}
                int_field: {int_field}
                float_field: {threshold}
            """)
        #do some processing on the image, in this example I just invert it
        if (logits_per_image.item() /100.0 > threshold):
            image = 0.0 * image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SafetyFilter": SafetyFilter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SafetyFilter": "Safety Filter"
}
