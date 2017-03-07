import helpful_functions

CATEGORIES = ['anemone', 'branches', 'flags', 'flowers', 'grass', 'smoke', 'water']
IMAGE_DIRECTORY = '/media/hdd/ben/dyntex_train_images/'
TEST_DIRECTORY = '/media/hdd/ben/dyntex_test_images/'
FRAMES_TO_WAIT = 10
IMAGE_SIZE = 128
IMAGES_PER_FRAME = 50
NUM_IMAGES = helpful_functions.get_num_images(IMAGE_DIRECTORY)
NUM_CATEGORIES = helpful_functions.get_num_categories()
NUM_TEST_IMAGES = helpful_functions.get_num_images(TEST_DIRECTORY)
TRAIN_TO_TEST_RATIO = 5
FPS = 30.0
IMAGES_PER_CHUNK = 10