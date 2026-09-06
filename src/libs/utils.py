import datetime
import os
import json
import requests
import shutil
import sys

import boto3
import litellm
import numpy as np
import openai
import piexif
import wikipedia

from PIL import Image, ImageDraw, ImageOps, ImageFont, ExifTags
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from src.utils import context


config = context.config
logger = context.logger
pipeline_start_time = datetime.datetime.now()


def get_dataset_files(DATA_DIR: str = context.root_dir + config.directories.data_input):
    """
    List all the relevant input files in the directory
    :param DATA_DIR:
    :return:
    """
    list_extensions = ('.jpeg', '.jpg', '.png')
    if os.path.exists(DATA_DIR):
        dir_files = os.listdir(DATA_DIR)
        file_paths = [os.path.join(DATA_DIR, i_dir_files) for i_dir_files in dir_files]
        file_paths = [file_path for file_path in file_paths if file_path.endswith(list_extensions)]
    else:
        logger.error(f"No files found matching extensions: {list_extensions}!")
        file_paths = []

    logger.info(f"Found {len(file_paths)} files in directory {DATA_DIR}")

    return file_paths


def get_wiki_image(search_term):
    """
    Retrieve Wikipedia endpoints for certain bird species
    :param search_term:
    :return:
    """

    logger.info(f"Attempting to retrieve wiki image for: {search_term}")

    WIKI_REQUEST = config.endpoints.wiki_request
    try:
        result = wikipedia.search(search_term, results=1)
        wikipedia.set_lang('en')
        wkpage = wikipedia.WikipediaPage(title=result[0])
        title = wkpage.title
        response = requests.get(WIKI_REQUEST+title)
        json_data = json.loads(response.text)
        img_link = list(json_data['query']['pages'].values())[0]['original']['source']

        logger.info(f"Successfully retrieved wiki image: {img_link}")

    except Exception as e:
        logger.error(f"Unable to retrieve wiki image for: {search_term}: {e}")
        img_link = ''

    return img_link


def get_filetimestamp(file_path):
    """
    Extract timestamp from a wav recording file name
    :param file_path:
    :return:
    """
    file_timestamp = file_path.split('/')[-1][:8]
    try:
        file_timestamp = int(file_timestamp)
    except Exception as e:
        logger.warning(f"Could not extract file timestamp from filename: {e}")
        file_timestamp = ''
    return file_timestamp


def gpt_app(message: str = None):
    """
    Execute ChatGPT Query
    :param message: sample 'Please provide a description for the veritree'
    :return:
    GPT response
    """

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[{'role': 'user',
                   'content': f'{message}'}]
    )

    output = response['choices'][0]['message']['content']

    return output


def sns_template(Run_Date):
    subject = "veritree App Notification System"
    message = """
        App Process Completed.

        ---------------------------------------------------------------------------------------------------------------
        Summary of App
        ---------------------------------------------------------------------------------------------------------------
        {a1:<20}  :  {a2}
        ---------------------------------------------------------------------------------------------------------------
        """.format(a1='Run Date', a2=Run_Date)

    return subject, message


def sns_email_notification(subject, message):
    """
    Sends an email notification to preconfigured SNS topic
    """
    try:
        sns_client = boto3.client(
            'sns',
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name = os.getenv("AWS_S3_REGION_NAME")
        )

        # Publish the message to the SNS topic
        response = sns_client.publish(
            TopicArn=config.sns.topic_arn,
            Subject=subject,
            Message=message
        )

        logger.info(f"Message sent! Message ID: {response['MessageId']}")

    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure your credentials.")
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials. Please check your credentials.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def setup_paths(imagedir_path, results_dir):
    """
    Creates a results directory and subdirectories for each image,
    and returns a list of image absolute paths

    Parameters
    ----------
    image_path: str
        path to image or image directory you want to run pipeline on

    results_dir: str
        path to directory in which to save pipeline results. If not
        provided, will be added to image directory

    Returns
    -------
    image_paths: list
        list of absolute  paths to individual images

    result_dir: str
        directory in which to save pipeline results
    """
    print(imagedir_path)
    image_paths = []
    # setting up result directory for the single image and
    # image directory options
    if results_dir:
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
    # else:
    # 	save_dir = imagedir_path
    if os.path.isdir(imagedir_path): # checking if directory
        image_names = os.listdir(imagedir_path)
        if not results_dir:
            results_dir = os.path.join(imagedir_path, 'results')
        if '.DS_Store' in image_names:
            image_names.remove('.DS_Store')
        for name in image_names:
            # checking image extensions
            ext_fine, file_id = check_image_extension(name)
            if not ext_fine:
                continue
            image_path = os.path.join(imagedir_path, name)
            image_paths.append(image_path)
            # making result directories for each image
            image_resultpath = os.path.join(results_dir, file_id)
            if not os.path.exists(image_resultpath):
                os.mkdir(image_resultpath)
    else:
        if not results_dir:
            outer_dir = '/'.join(imagedir_path.split('/')[0:-1])
            results_dir = os.path.join(outer_dir, 'results')
        if os.path.exists(results_dir): # removing previous dirs
            shutil.rmtree(results_dir)
        os.mkdir(results_dir)
        # checking image extensions
        # TODO: name variable
        name = ''
        ext_fine, file_id = check_image_extension(name)
        if ext_fine:
            image_resultpath = os.path.join(results_dir, file_id)
            # making result directory for image
            image_paths.append(imagedir_path)
            if not os.path.exists(image_resultpath):
                os.mkdir(image_resultpath)

    # early exit option if no acceptable images supplied
    if len(image_paths) == 0:
        logger.info('No suitable images supplied. Exiting...')
        shutil.rmtree(results_dir)
        sys.exit()

    return image_paths, results_dir


def plot_boxes(image, boxes, class_conf, onehot_to_str, path_save, legend=True):
    """
    Plots object detection image with bounding boxes.
    Box outlines are colored to match predicted survival status

    Parameters
    ----------
    save_path: str
        path to save plot to

    image: PIL object
        object detection image to draw boxes on

    boxes: list
        object detection bounding boxes in YOLO format

    class_conf: list
        survival status onehot confidence scores

    onehot_to_str: dict
        mapping from onehot labels to string labels

    Returns
    -------
    None
    """

    draw = ImageDraw.Draw(image)
    pix_width, pix_height = image.size
    # scaling fontsize to image size
    fontsize = int(np.sqrt(pix_width*pix_height)/50)
    label_fontsize = int(np.sqrt(pix_width*pix_height)/40)

    try:
        font = ImageFont.truetype('/Library/Fonts/Andale Mono.ttf', fontsize)
        label_font = ImageFont.truetype('/Library/Fonts/Andale Mono.ttf', label_fontsize)
    except Exception as e:
        logger.error(f"Unable to load PIL fonts: {e}")
        font = ImageFont.load_default(size=fontsize)
        label_font = ImageFont.load_default(size=label_fontsize)

    # box colors in legend
    color_legend = {'Unclear':      (97, 155, 242),
                    'Dead/dormant': (250, 107, 107),
                    'Alive':        (55, 204, 39)}

    for idx, box in enumerate(boxes):
        box_coords = get_box_coords(image, box, pad_fraction=0, border=0)
        # getting string label from onehot score
        score = [float(x) for x in class_conf[idx]] # ===>> [0.124763645, 0.42603365, 0.44920275]
        template_array = np.zeros(len(onehot_to_str))
        template_array[np.argmax(score)] = 1.0
        label = onehot_to_str[str(list(template_array))]
        color = color_legend[label]
        # scaling box linewidth to image size
        linewidth = int(np.sqrt(pix_width*pix_height)/250)
        # drawing box for label
        textsize = draw.textlength(str(idx), font)
        box_size = textsize+10
        if box_coords['left'] - box_size < 0:
            left = box_coords['right']
            right = box_coords['right']+box_size
            top = box_coords['bottom']+fontsize*1.5
            bottom = box_coords['bottom']
        else:
            left = box_coords['left']-box_size
            right = box_coords['left']
            top = box_coords['bottom']+fontsize*1.5
            bottom = box_coords['bottom']
        draw.rectangle([left, bottom, right, top], fill=color)
        # labeling box
        draw.text((left+0.05*fontsize, bottom), str(idx), (0,0,0), font=label_font)
        # draw black outline behind rectangle
        draw.rectangle([box_coords['left']-2, box_coords['bottom']-2, box_coords['right']+2, box_coords['top']+2], outline='black', width=linewidth+4)
        # drawing box on image
        draw.rectangle([box_coords['left'], box_coords['bottom'], box_coords['right'], box_coords['top']], outline=color, width=linewidth )

    if legend:
        # making legend (also scaled with image size)
        legend_dim = [0, pix_height-fontsize*4.2, fontsize*11, pix_height]
        draw.rectangle(legend_dim, fill='white', outline='black')
        # filling out legend (future fix: this does not scale with the length of the labels)

        for idx, key in enumerate(color_legend):
            txt = key.ljust(13) + ': '
            draw.text((fontsize/3, pix_height-1.3*fontsize*(idx+1)), txt, (0,0,0), font=font)
            rec_dim = [9.5*fontsize, pix_height-1.3*fontsize*(idx+1), 10.5*fontsize, pix_height-1.3*fontsize*(idx+1)+fontsize]
            draw.rectangle(rec_dim, fill = color_legend[key])

    if path_save:
        # save annotated image
        image.save(path_save)

    return image


def save_data(data, str_to_onehot, save_path):
    """
    Takes dictionary of pipeline results, compiles them into an array, saves array as csv

    Parameters
    ----------
    data: dict
        dictionary containing coarse pipeline results (AKA
        number of mangroves detected and number in each class)

    str_to_onehot: dict
        mapping from string labels to onehot labels

    save_path: str
        path to save array to as csv

    Returns
    -------
    csv_data: list
        list of coarse pipeline results
    """
    # building csv of number of detected mangroves and their classes
    csv_data = [['ID', 'detected_mangroves', 'Alive', 'Dead/dormant', 'Unclear']]
    for idx, id in enumerate(data['ids']):
        class_counts = np.zeros(len(str_to_onehot))
        for score in data['class_conf'][idx]:
            score = [float(x) for x in score]
            template_array = np.zeros(len(str_to_onehot))
            template_array[np.argmax(score)] = 1.0
            class_counts += template_array
        csv_data.append([id, len(data['boxes'][idx]), int(class_counts[0]), int(class_counts[1]), int(class_counts[2])])

    # saving results
    np.savetxt(save_path, csv_data, delimiter=',', fmt='%s')

    return csv_data


def safe_listdir(path):
    """ Listdir that ignores hidden files """
    safe_list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            safe_list.append(f)

    return safe_list


def rotate_image(image, image_path, save_rotation=False):
    """
    Rotates image to match its exif information. Resets exif information after

    Parameters
    ----------
    image : PIL object
        PIL image object to be rotated

    Returns
    -------
    image: PIL object
        PIL image object after rotating

    """
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    try:
        exif = image._getexif()
        # Get the orientation if it exists
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

    except:
        logger.warning('No exif information available for ' + '{}'.format(image.filename))
        orientation = None

    if save_rotation:
        image.save(image_path, exif=exif)

    return image


def crop_image(image, bbox, border, pad_fraction, min_size):
    """
    Crops image using YOLO bounding box and additional arguments

    Parameters
    ----------
    image : PIL object
        image to be cropped

    bbox: list
        YOLO fractional bounding box coordinates

    border: int
        size of zero-padding border around image

    pad_fraction: float
        fraction of image dimensions in pixels to extend crop boundaries

    min_size: int
        minimum image side length to resize up to

    Returns
    -------
    cropped_image: PIL object
        PIL image object, cropped to fit the bounding box plus fractional padding

    """
    pix_width = image.size[0] - 2 * border
    pix_height = image.size[1] - 2 * border
    # pix_width, pix_height = image.size
    box_coords = get_box_coords(image, bbox, pad_fraction, border)
    # resizing image if width too small
    if abs(box_coords['right'] - box_coords['left']) < min_size:
        box_coords['left'] = round(pix_width * (box_coords['x_center']) - min_size + border)
        box_coords['right'] = round(pix_width * (box_coords['x_center']) + min_size + border)
    # resizing image if height too small
    if abs(box_coords['top'] - box_coords['bottom']) < min_size:
        box_coords['bottom'] = round(pix_height * (box_coords['y_center']) - min_size + border)
        box_coords['top'] = round(pix_height * (box_coords['y_center']) + min_size + border)

    cropped_image = image.crop((box_coords['left'], box_coords['bottom'], box_coords['right'], box_coords['top']))

    return cropped_image


def get_box_coords(image, label, pad_fraction, border):
    """
    Returns center and corner coordinates for bounding box

    Parameters
    ----------
    image: PIL object
        image bounding box was predicted from

    label: list
        bounding box coordinates in YOLO fractional format

    pad_fraction: float
        fraction of box dimensions to extend cropping
        boundaries by

    border: int
        zero padding used around PIL image

    Returns
    -------
    box_coords: dict
        dictionary containing box center and corner coordinates
    """
    box_coords = {}
    pix_width = image.size[0] - 2 * border
    pix_height = image.size[1] - 2 * border
    # pix_width, pix_height = image.size
    x_center = label[0]
    box_coords['x_center'] = x_center
    y_center = label[1]
    box_coords['y_center'] = x_center
    width = label[2]
    height = label[3]
    # getting box corner coordinates in pixels
    box_coords['left'] = round(pix_width * (x_center - width / 2 - pad_fraction * width) + border)
    box_coords['right'] = round(pix_width * (x_center + width / 2 + pad_fraction * width) + border)
    box_coords['bottom'] = round(pix_height * (y_center - height / 2 - pad_fraction * height) + border)
    box_coords['top'] = round(pix_height * (y_center + height / 2 + pad_fraction * height) + border)

    return box_coords


def create_mappings(class_names):
    """
    Returns one-hot and integer mappings to a list of class strings

    Parameters
    ----------
    class names: list
        list of class names corresponding to image labels

    Returns
    -------
    one_hot_map: dict
        dictionary with class names as keys and one-hot mappings as values

    integer_map: dict
        dictionary with class names as keys and integer mappings as values

    """
    one_hot_map = {}  # mapping string labels to onehot labels
    for idx, label_type in enumerate(class_names):
        template_array = np.zeros(len(class_names))
        template_array[idx] = 1
        one_hot_map[label_type] = template_array

    integer_map = {}  # mapping string labels to int labels
    for idx, label_type in enumerate(class_names):
        integer_map[label_type] = idx

    return one_hot_map, integer_map


def decode_one_hot(predictions, one_hot_map, integer_map):
    """
    Converts list of one-hot labels to their corresponding string and integer labels

    Parameters
    ----------
    predictions: list
        list of one-hot labels

    one_hot_map: dict
        mapping of class strings to one-hot values

    integer_map: dict
        mapping of class strings to integer values

    Returns
    -------
    str_labels: list
        list of class strings corresponding to predictions variable

    int_labels: list
        list of class integer corresponding to predictions variable

    """
    max_pred = []
    for i in predictions:
        template_array = np.zeros(len(one_hot_map))
        template_array[np.argmax(i)] = 1
        max_pred.append(str(template_array))
    onehot_to_str = {str(j): i for i, j in one_hot_map.items()}
    str_labels = np.array([onehot_to_str[x] for x in max_pred])

    int_labels = np.array([integer_map[x] for x in str_labels])

    return str_labels, int_labels


def check_image_extension(name):
    """ Checks if name has right extension. Returns boolean and
    image name without extension """
    # expected image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png']
    file_id, file_ext = os.path.splitext(name)
    if file_ext not in supported_extensions:
        logger.info(f'Image with {file_ext} does not have one of the following '
                    f'supported extensions: {supported_extensions}. Skipping...')
        ext_fine = False
    else:
        ext_fine = True

    return ext_fine, file_id


def cast_classes(classes_yaml):
    """
    Transform classes defined in config.yaml to string mappings
    :return:
    """
    survivability_dict = dict(classes_yaml)
    str_to_onehot = {x: survivability_dict[x][0] for x in survivability_dict}
    onehot_to_str = {str(j): i for i, j in str_to_onehot.items()}

    return str_to_onehot, onehot_to_str


def transform_img_dimensions(cropped_detected_mangrove, trans_width, trans_height):
    # TODO: replace config parameters with the function definitions
    # check if width/height equal
    if cropped_detected_mangrove.size[0] == cropped_detected_mangrove.size[1]:
        cropped_detected_mangrove = cropped_detected_mangrove.resize(
            (config.pipeline_meta.input_width, config.pipeline_meta.input_height))
    else:
        small_idx = np.argmin(cropped_detected_mangrove.size)
        large_idx = np.argmax(cropped_detected_mangrove.size)

        # getting resizing dimensions
        new_small_length = int(config.pipeline_meta.input_width * (
                    cropped_detected_mangrove.size[small_idx] / cropped_detected_mangrove.size[large_idx]))
        new_size = [0, 0]
        new_size[small_idx] = new_small_length
        new_size[large_idx] = config.pipeline_meta.input_width
        cropped_detected_mangrove = cropped_detected_mangrove.resize(tuple(new_size))

    # padding rest of image with zeros to be equal to input_height
    cropped_detected_mangrove = ImageOps.pad(cropped_detected_mangrove,
                                             (config.pipeline_meta.input_width, config.pipeline_meta.input_height))

    return cropped_detected_mangrove


def empty_folder(folder=''):
    """
    Function deletes all files from a given 'folder'
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Unable to remove {file_path}: {e}')

    return None


def delete_file(file_path):
    """

    :param file_path:
    :return:
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"File deleted: {file_path}")
    else:
        logger.warning(f"File not found: {file_path}")


def check_file_exists(filepath, path_type='local'):
    """
    Check if the file exists
    """

    if path_type == 'local':
        bool_result = os.path.exists(filepath)

    elif path_type == 's3':
        raise NotImplemented
        # TODO: validate path type from AWS index
        # bool_result = filepath in awspaths

    return bool_result


def copy_exif(source_path, target_path, output_path):
    """
    Function to preserve EXIF data from source to target path.
    :param source_path: source image to copy EXIF data from
    :param target_path: target image to paste EXIF data to
    :param output_path: output path to write new image to
    :return:
    """
    source_img = Image.open(source_path)
    exif_dict = piexif.load(source_img.info.get("exif", b""))
    target_img = Image.open(target_path)
    exif_bytes = piexif.dump(exif_dict)
    target_img.save(output_path, exif=exif_bytes)

    return None


PROVIDER_ALIASES = {
    'gpt': 'gpt-4o',
    'claude': 'anthropic/claude-sonnet-4-5',
    'gemini': 'gemini/gemini-2.5-flash',
}


def llm_chat(system_prompt: str, user_prompt: str, model: str = 'gpt-4o', temperature: float = 0.2) -> str:
    """
    Route LLM chat to the correct provider.
    Accepts short aliases ('gpt', 'claude', 'gemini') or full model names.
    """

    if model in PROVIDER_ALIASES:
        litellm_model = PROVIDER_ALIASES[model]
    elif model.startswith('gemini-'):
        litellm_model = f"gemini/{model}"
    elif model.startswith('claude-'):
        litellm_model = f"anthropic/{model}"
    elif model.startswith('gpt-') or model.startswith('o1-') or model.startswith('o3-'):
        litellm_model = model
    else:
        raise ValueError(f"Unknown model '{model}'. Use an alias ('gpt', 'claude', 'gemini') or a full model name with a recognised prefix.")

    if 'anthropic' in litellm_model:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        key_name = "ANTHROPIC_API_KEY"
    elif 'gemini' in litellm_model:
        api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        key_name = "GOOGLE_AI_STUDIO_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        key_name = "OPENAI_API_KEY"

    if not api_key:
        raise ValueError(f"API key not set. Please set {key_name} in your environment.")

    logger.info(f"llm_chat: alias='{model}' → model='{litellm_model}'")

    response = litellm.completion(
        model=litellm_model,
        api_key=api_key,
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ],
    )
    return response.choices[0].message.content