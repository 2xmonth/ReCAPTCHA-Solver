from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from colorama import Style, Fore, init
import undetected_chromedriver as uc
import selenium.common.exceptions
from selenium import webdriver
from datetime import datetime
from time import sleep
import pyautogui
import random
import torch
import os

from improve import improve, load_model
from curves.humanclicker import HumanClicker

init(autoreset=True)
hc = HumanClicker()

problem_captchas = {"bicycle", "stairs", "chimney", "boat"}  # i havent even added all of them to the list below. you can find more of these in the recaptcha__en.js file
big_problem_captchas = {"mountains or hills", "stop signs", "footpaths", "sidewalks", "street signs", "storefront", "grass", "shrubs", "bodies of water", "such as lakes or oceans", "cactus", "rivers", "limousines", "houses"}


def extract_images(driver):
    sleep(0.5)
    image_elements = driver.find_elements(By.CLASS_NAME, "rc-image-tile-wrapper")

    return image_elements


def extract_prompt(driver):
    prompt = driver.find_element(By.XPATH, "/html/body/div/div/div[2]/div[1]/div[1]/div/strong")

    return prompt.text


def get_driver(type):
    """
    "geckodriver" or "chromedriver"
    Returns the driver object
    """
    if type == "chromedriver":
        driver = uc.Chrome()

    else:
        driver = webdriver.Firefox()

    driver.maximize_window()

    return driver


def compare(l1, l2):
    if len(l1) != len(l2):
        return False
    set1 = set(l1)
    set2 = set(l2)

    if set1 == set2:
        return True
    else:
        return False


def load_models():
    print(f"{Style.BRIGHT}{Fore.YELLOW}Loading image-rec model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', verbose=False)
    model.conf = 0.15
    print(f"{Style.BRIGHT}{Fore.GREEN}Loaded image-rec model")

    super_res_model = load_model("scunet_color.pth")

    return model, super_res_model


def new_captcha(driver):
    actions = ActionChains(driver)

    reload = driver.find_element(By.ID, "recaptcha-reload-button")
    actions.move_to_element(reload).perform()
    sleep(0.1)
    actions.click().perform()
    print(f"{Style.BRIGHT}{Fore.YELLOW}Requested new captcha")


def error_handling(error, driver, completed_fading=None):
    print(f"{Style.BRIGHT}{Fore.RED}Error: {error}")

    if "Please try again" in error:
        base(driver)

    elif "Please select all matching images" in error:  # model didnt find all the images
        new_captcha(driver)
        base(driver)

    elif "Please also check the new images" in error:  # model didnt find any images in the new images
        if completed_fading:
            new_captcha(driver)
            base(driver)

        else:
            print(f"{Style.BRIGHT}{Fore.RED}Model did not finish the fading captcha, retrying...")
            base(driver)

    else:
        print(f"{Style.BRIGHT}{Fore.RED}Unknown Error: {error}")


def extract_token(driver):
    token = driver.find_element(By.XPATH, "//*[@id=\"recaptcha-token\"]").get_attribute("value")  # inputs dont have inner html so you have to do it like this. (stackoverflow.com/questions/48846700)

    return token


def load(driver):
    global last_point

    driver.get("https://google.com/recaptcha/api2/demo")
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/form/fieldset/legend")))

    iframe = driver.find_element(By.XPATH, "/html/body/div[1]/form/fieldset/ul/li[5]/div/div/div/div/iframe")
    driver.switch_to.frame(iframe)

    check = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[1]/div/div/span/div[1]")
    check_element_width = int(check.size["width"] / 2)  # int is for floating points
    check_element_height = int(check.size["height"] / 2)

    check_element_x = check.location["x"] + 28 + check_element_width  # updated because the pos was wrong
    check_element_y = check.location["y"] + 438 + check_element_height  # ^

    random_start = (random.randint(199, 399), random.randint(99, 599))
    end = (check_element_x, check_element_y)

    points = hc.get_points(start=random_start, end=end, offsetBoundaryX=20, offsetBoundaryY=20, knotCounts=0)  # 24x24 button size
    last_point = end

    for point in points:
        pyautogui.moveTo(point)

    pyautogui.click()
    driver.switch_to.default_content()
    iframe = driver.find_element(By.XPATH, "/html/body/div[2]/div[4]/iframe")
    driver.switch_to.frame(iframe)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//*[@id=\"rc-imageselect\"]")))


def solve(image_elements, prompt, driver):
    sleep(0.5)

    if "vehicle" in prompt:
        prompt = {"car", "truck", "bus", "bicycle", "motorcycle", "tractor", "boat"}

    cwd = os.getcwd()

    new_detections = []
    elements = []

    try:
        for i, element in enumerate(image_elements, start=1):
            element.screenshot(f"{cwd}/{i}.png")
            improve(super_res_model, f"{i}.png", result_dir=os.getcwd())
            result = model(f"{i}_result.png", size=128)
            # result2 = model(image_before, size=128)
            os.remove(f"{i}_result.png")
            os.remove(f"{i}.png")

            dataframe = result.pandas().xyxy[0]

            if dataframe.empty:  # dataframe will be empty if there are no detections
                continue

            # dataframe = result2.pandas().xyxy[0]
            #
            # if dataframe.empty:  # dataframe will be empty if there are no detections
            #     continue

            detections = dataframe.loc[:, "name"].values
            print(f"{Style.BRIGHT}{Fore.MAGENTA}{i} {detections}")
            # print(f"{Style.BRIGHT}{Fore.MAGENTA}{i} {detections} non super res")

            for result in detections:
                new_detections.append(result)
                if result in prompt:
                    print(f"{Style.BRIGHT}{Fore.GREEN}Image {i}_result.png")

                    elements.append(element)

                    break

    except selenium.common.exceptions.StaleElementReferenceException:
        print(f"{Style.BRIGHT}{Fore.RED}Ran into a stale element while solving. Retrying...")  # problem with the fading captchas
        image_elements = extract_images(driver)
        solve(image_elements, prompt, driver)

    return new_detections, elements


def base(driver, prev_detections=None):
    global completed_fading, last_point

    image_elements = extract_images(driver)
    prompt = extract_prompt(driver)

    if prompt in big_problem_captchas:
        print(f"{Style.BRIGHT}{Fore.RED}\"{prompt}\" is not solvable. Retrying...")
        new_captcha(driver)
        base(driver)

    if prompt in problem_captchas:
        print(f"{Style.BRIGHT}{Fore.RED}Alert! \"{prompt}\" is difficult to solve.")

    new_detections, elements = solve(image_elements, prompt, driver)

    random.shuffle(elements)

    for element in elements:
        end_element_width = int(element.size["width"] / 2)
        end_element_height = int(element.size["height"] / 2)

        end_element_x = element.location["x"] + 48 + end_element_width
        end_element_y = element.location["y"] + 178 + end_element_height

        end = (end_element_x, end_element_y)

        points = hc.get_points(start=last_point, end=end, offsetBoundaryX=96, offsetBoundaryY=96, knotsCount=1)  # 96 for the 4x4 captchas
        last_point = end
        for point in points:
            pyautogui.moveTo(point)

        pyautogui.click()

        sleep(0.2)

    sleep(0.25)

    try:
        try:
            check_element = driver.find_element(By.XPATH, "//*[@id=\"recaptcha-anchor\"]").get_attribute("aria-checked")
            if check_element == "true":
                print(f"{Style.BRIGHT}{Fore.GREEN}Solved captcha, extracting the token")
                sleep(0.25)
                raise Finished()

        except selenium.common.exceptions.NoSuchElementException:
            check_element = False

        type_element = driver.find_element(By.XPATH, "/html/body/div/div/div[2]/div[1]/div[1]/div/span").text
        if "Click verify once there are none left" in type_element and prev_detections is None:
            print(f"{Style.BRIGHT}{Fore.YELLOW}Fading images detected. Setting bogus values")
            prev_detections = ["bogus to start the detections"]

    except selenium.common.exceptions.NoSuchElementException:
        print(f"{Style.BRIGHT}{Fore.YELLOW}Couldn't find a type_element")

    if prev_detections is not None:
        if compare(prev_detections, new_detections):
            completed_fading = True

        else:
            sleep(1)
            completed_fading = False
            base(driver, new_detections)

    sleep(0.75)

    button = driver.find_element(By.ID, "recaptcha-verify-button")

    button_width = int(button.size["width"] / 2)
    button_height = int(button.size["height"] / 2)

    button_x = button.location["x"] + 48 + button_width
    button_y = button.location["y"] + 178 + button_height  # 178 here to make it go 30 pixels down extra

    button = (button_x, button_y)

    points = hc.get_points(start=last_point, end=button, knotsCount=0, offsetBoundaryX=90, offsetBoundaryY=35, distortionFrequency=0.0)  # 96x40 button size
    last_point = button
    for point in points:
        pyautogui.moveTo(point)

    pyautogui.click()

    sleep(0.5)

    # error handling
    try:
        error1 = driver.find_element(By.CLASS_NAME, "rc-imageselect-incorrect-response")
        error2 = driver.find_element(By.CLASS_NAME, "rc-imageselect-error-select-more")
        error3 = driver.find_element(By.CLASS_NAME, "rc-imageselect-error-dynamic-more")

        if error1.get_attribute("tabindex") == "0":
            error_handling(error1.text, driver)

        elif error2.get_attribute("tabindex") == "0":
            error_handling(error2.text, driver, completed_fading=completed_fading)

        elif error3.get_attribute("tabindex") == "0":
            error_handling(error3.text, driver, completed_fading=completed_fading)

    except (selenium.common.exceptions.NoSuchElementException, selenium.common.exceptions.NoSuchAttributeException) as e:
        if e == selenium.common.exceptions.NoSuchElementException:
            print(f"{Style.BRIGHT}{Fore.YELLOW}Error elements not found?")

        elif e == selenium.common.exceptions.NoSuchAttributeException:  # if it cant find the attribute. NORMAL if the error didnt appear
            pass

    driver.switch_to.default_content()

    iframe = driver.find_element(By.XPATH, "/html/body/div[1]/form/fieldset/ul/li[5]/div/div/div/div/iframe")
    driver.switch_to.frame(iframe)

    sleep(0.3)

    try:
        check_element = driver.find_element(By.XPATH, "//*[@id=\"recaptcha-anchor\"]").get_attribute("aria-checked")
        if check_element == "true":
            print(f"{Style.BRIGHT}{Fore.GREEN}Solved captcha, extracting the token")
            sleep(0.25)
            raise Finished()

        elif check_element == "false":
            check_element = False

    except selenium.common.exceptions.NoSuchElementException:
        check_element = False

    if check_element is False:
        print(f"{Style.BRIGHT}{Fore.YELLOW}Captcha skipped without errors, continuing...")
        driver.switch_to.default_content()
        iframe = driver.find_element(By.XPATH, "/html/body/div[2]/div[4]/iframe")
        driver.switch_to.frame(iframe)
        base(driver)


class Finished(Exception):  # i know this is a bad way to do it, but it works
    pass


def main(driver_type):
    driver = get_driver(driver_type)

    load(driver)
    try:
        base(driver)
        assert False, "should never reach here"

    except Finished:
        token = extract_token(driver)
        return token


logo = f"""{Style.BRIGHT}{Fore.GREEN}

                                                                                                        ░░░░░░  ░░░░░░                                          
                                                                                            ░░        ░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░                                        
                                                                              ░░░░░░░░░░▒▒▒▒░░░░  ░░  ░░▓▓▒▒▓▓▓▓▓▓▓▓▒▒▒▒░░░░                                    
                                                                            ░░░░▒▒░░░░▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▓▓▓▓▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒░░                                  
                                                                          ░░▒▒▓▓▒▒░░▒▒▓▓▓▓▓▓░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓░░                              
                                                                        ░░▒▒▒▒▓▓▒▒░░▒▒▓▓▓▓▒▒░░▒▒▒▒▒▒▒▒▓▓▒▒▒▒▓▓██▓▓▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒░░                          
                                                                  ░░░░  ▒▒▓▓▓▓▒▒▒▒▒▒▒▒██▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▒▒▒▒░░                        
                                                              ░░  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒                        
                                                            ░░▒▒▒▒▒▒▓▓▒▒▒▒░░▒▒▓▓████▓▓▒▒▒▒▓▓██▓▓▒▒▓▓▒▒▒▒▓▓▓▓▓▓▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒██▓▓▓▓▓▓▓▓▒▒                      
                                                          ░░▒▒░░▒▒▒▒▒▒░░▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▓▓▒▒▓▓██▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒▓▓▒▒▒▒▒▒▒▒░░                    
                                                        ▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▓▓██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▒▒                    
                                                      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒                    
                                                      ▒▒▒▒▒▒▒▒▓▓▓▓▒▒░░▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▓▓▒▒██▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▓▓▓▓▒▒▓▓▒▒                    
                                                      ▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒▓▓▒▒▒▒▒▒░░▒▒▓▓▒▒▒▒▓▓▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                  
                                                      ▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▓▓██▓▓▒▒░░░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒░░                
                                                      ░░▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓░░░░▒▒▒▒▓▓▒▒▒▒▒▒▒▒▓▓██▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒                
                                                    ▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒░░▓▓▓▓▓▓▓▓▓▓▒▒▓▓▒▒▒▒▒▒▒▒░░▓▓▒▒▒▒▒▒▒▒░░▒▒▒▒▓▓▒▒▓▓▓▓▓▓▒▒▒▒░░            
                                                ░░▒▒░░▒▒▒▒▒▒▓▓▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒░░▒▒▒▒▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒            
                                                ░░▒▒▒▒▒▒▒▒▓▓▒▒▒▒▓▓▓▓▓▓▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓██▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▓▓▒▒▓▓▒▒▒▒▒▒▒▒░░            
                                                ░░▒▒▒▒▓▓▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░      
                                          ░░░░  ▒▒░░░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░▒▒▒▒▒▒▓▓▓▓▒▒░░▒▒░░▒▒▒▒░░▒▒▓▓▓▓▒▒▒▒░░▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒▒▓▓▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒░░    
                                        ░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▓▓▓▓▒▒▒▒░░▓▓▓▓▒▒▒▒▓▓▒▒▒▒▒▒██▒▒░░▒▒▒▒░░▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓██▓▓▓▓▒▒▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░▒▒░░  
                                        ▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▓▓▓▓▓▓▓▓▓▓██▒▒██▓▓▓▓░░▒▒▒▒▓▓██▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒░░
                                        ▒▒▒▒▓▓▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▓▓▒▒▒▒▒▒▓▓▒▒████▒▒▒▒▒▒▓▓▓▓▒▒  
                                        ░░▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒██▓▓▒▒▒▒▓▓▓▓▒▒▒▒▒▒▓▓▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                        ▒▒▒▒░░▒▒▒▒▓▓██▒▒██▓▓▓▓▒▒▓▓▓▓▒▒▓▓████▒▒██▓▓▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒░░▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▓▓▒▒▓▓▓▓▒▒▓▓▒▒▒▒▓▓██▓▓▒▒
                                          ░░░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▒▒▓▓▒▒▒▒▒▒▓▓██▓▓▒▒▒▒▒▒██▓▓▒▒▓▓▒▒▓▓▒▒▓▓▒▒▓▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▒▒▒▒▓▓▓▓▒▒▒▒▓▓▒▒▒▒▒▒▒▒  
                                              ▒▒░░▒▒▒▒░░░░▓▓▓▓▓▓▒▒░░▓▓▒▒░░▓▓▓▓██▒▒██▓▓▓▓▒▒▒▒  ▓▓▓▓▓▓░░▓▓▓▓▓▓▒▒▓▓▓▓░░▓▓▓▓  ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒    
                                              ░░░░░░▓▓▒▒▓▓▒▒██▒▒▓▓▓▓░░▒▒  ▒▒██▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▒▒▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓░░    ░░░░▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒░░    
                                                    ▒▒        ░░  ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓  ░░▓▓▓▓    ░░░░░░▒▒▒▒██▒▒▒▒▒▒▒▒▒▒▓▓▒▒▓▓▒▒▒▒░░    
                                                                            ░░▒▒▒▒        ░░▓▓▓▓██▓▓▓▓░░  ▓▓░░    ░░          ▒▒▒▒▒▒▓▓▒▒░░▒▒▒▒▒▒  ▒▒▒▒▒▒        
                                                                              ░░              ▒▒▓▓██▓▓░░▓▓▒▒                  ▒▒▒▒▒▒▒▒      ░░▒▒                
                                                                                                ▒▒██▓▓▓▓░░                    ░░░░  ░░                          
                                                                                                ░░▓▓██▓▓                                                        
                                                                                                ░░▓▓▓▓▒▒                                                        
                                                                                                ░░▓▓▓▓▒▒                                                        
                                                                                                ░░▓▓▓▓▓▓                                                        
                                                                                                ▒▒▓▓▓▓▓▓                                                        
                                                                                                ▓▓▓▓▓▓▓▓                                                        
                                                                                            ░░▓▓▓▓▓▓▓▓▓▓▒▒                                                      
"""

banner = f"""{Style.BRIGHT}{Fore.GREEN}                                                                                                  
                ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
                ║╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗     ╔══════════════════════════╗     ╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗╔╗║
                ╠╣╠╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬═════╣ By: t.me/vind1ctus v2.0  ╠═════╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╣╠╬╬╣         
                ║╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝     ╚══════════════════════════╝     ╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝║
                ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

"""
last_point = None
os.system("clear")

print(f'\33]0;ReCAPTCHA Solver v2\a', end='', flush=True)

print(logo)
print(banner)

print(f"{Style.BRIGHT}{Fore.CYAN}Driver type? (geckodriver, chromedriver) ===>> ", end="")
driver_type = input()

if driver_type != "geckodriver" and driver_type != "chromedriver":
    print(f"{Style.BRIGHT}{Fore.YELLOW}Invalid driver. Defaulting to chromedriver")
    driver_type = "chromedriver"

model, super_res_model = load_models()

sleep(2)

os.system("clear")
start_time = datetime.now()

token = main(driver_type)

elapsed = datetime.now() - start_time
elapsed = str(elapsed)[:-7]

print(f"{Style.BRIGHT}{Fore.GREEN}Extracted ReCAPTCHA token: {token}")
print(f"\n{Style.BRIGHT}{Fore.CYAN}Finished in: {elapsed}")
