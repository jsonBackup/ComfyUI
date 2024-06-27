from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import open_clip

class LatentSafetyFilter:
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
                "samples": ("LATENT", ),
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

    RETURN_TYPES = ("LATENT",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Safety"

    def get_model_info(self, model_ID, device):
        model = CLIPModel.from_pretrained(model_ID).to(device)
        processor = CLIPProcessor.from_pretrained(model_ID)
        tokenizer = CLIPTokenizer.from_pretrained(model_ID)
        return model, processor, tokenizer

    def test(self, samples, safety_filter, int_field, threshold, print_to_screen):
        models = {'B-8': {'model_name':'Latent-ViT-B-8-512',
                  'pretrained':'/dlabdata1/wendler/models/latent-clip-b-8.pt'},
          'B-4-plus':{'model_name':'Latent-ViT-B-4-512-plus',
                      'pretrained':'/dlabdata1/wendler/models/latent-clip-b-4-plus.pt'}}
        size = 'B-4-plus'
        model_name = models[size]['model_name']
        pretrained = models[size]['pretrained']
        model_latent, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer_latent = open_clip.get_tokenizer(model_name)

        image_features = model_latent.encode_image(samples["samples"])
        text_features = model_latent.encode_text(tokenizer_latent([f"an image of {safety_filter}", f"an image of no {safety_filter}"]))

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
        
        for i, sample in enumerate(samples["samples"]):


            if text_probs[i][0].item() > threshold:
                samples["samples"][i].zero_()
                print("Sample", i, "processed: Set to zero")
            else:
                print("Sample", i, "processed: Not set to zero")

            print("Probability:", text_probs[0][0].item())
            print("Threshold:", threshold)
            #print("Safety Filter:", safety_filters[i])

        return (samples,)
        if text_probs[0][0].item() > threshold:
            samples["samples"].zero_()
            print("THIS")
        else:
            print("NOT THIS")
        print(text_probs[0][0].item())
        print(threshold)
        print(safety_filter)
        return (samples, )
        #text_features = model_latent.encode_text(captions.cuda())
        print(image_features.shape)
        image_features_np = image_features.detach().numpy()
        text_features_np = text_features.detach().numpy()

        similarity_score = cosine_similarity(image_features_np, text_features_np)
        print(f"Similarity ({text}):\t{similarity_score}")




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
    "LatentSafetyFilter": LatentSafetyFilter

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSafetyFilter": "Latent Safety Filter"
}
