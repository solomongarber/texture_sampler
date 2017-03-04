import helpful_functions

CATEGORIES = ['anemone', 'branches', 'flags', 'flowers', 'grass', 'smoke', 'water']
IMAGE_DIRECTORY = './images/'
TEST_DIRECTORY = './static_test_images'
FRAMES_TO_WAIT = 10
IMAGE_SIZE = 128
IMAGES_PER_FRAME = 10
NUM_IMAGES = helpful_functions.get_num_images(IMAGE_DIRECTORY)
NUM_CATEGORIES = helpful_functions.get_num_categories()
NUM_TEST_IMAGES = helpful_functions.get_num_images(TEST_DIRECTORY)