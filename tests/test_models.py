from ..model_interfaces import *

if __name__ == "__main__":
    import pdb
    image = Image.open("/home/zhanxin/foundation_obj_nav/room_test3.png")

    gdino = VLM_GroundingDino()
    boxes, labels = gdino.detect_all_objects(image)
    print(boxes)
    print(labels)

    blip = VLM_BLIP()
    output = blip.query(image, "Which room is the photo?")
    print(output)

    folder_path = "/home/zhanxin/foundation_obj_nav/data/rls_tour_fisheye"

    # List all files in the folder
    files = os.listdir(folder_path) # file name list
    import matplotlib.pyplot as plt
    # Read and process each image
    for image_file in files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        output = blip.query(image, "Which room is the photo?")
        boxes, labels, _ = gdino.detect_all_objects(image)
        plt.imshow(image)
        ax = plt.gca()
        
        for i in range(len(labels)):
            label = labels[i]
            min_x, min_y, max_x, max_y = boxes[i]
            ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
            ax.text(min_x, min_y, label)
        plt.savefig(os.path.join(folder_path, image_file[:-4] + "_" + output + ".png") )
        plt.clf()


    llm_config_path="configs/gpt_config.yaml"
    llm = GPTInterface(config_path=llm_config_path)
    llm.check_goal('potted plant')
    
    role = "You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance on where to explore based on a series of observations. Observations will be given as descriptions of objects seen from four cameras in four directions. Your job is to estimate the robot's state. You will be given two descriptions, and you need to decide whether these two descriptions describe the same room. For example, if we have visited the room before and got one description, when we visit a similar room and get another description, it is your job to determine whether the two descriptions represent the same room. You should understand that descriptions may contain errors and noise due to sensor noise and partial observability. Always provide reasoning along with a deterministic answer. If there are no suitable answers, leave the space after 'Answer: None.' Always include: Reasoning: <your reasoning> Answer: <your answer>."
    # import re
    discript1 = "Description 1: On the left, there is wallhangingdecoration. On the right, there is nothing. In front of me, there is a white wood stool and a brown tile floor. Behind me, there is a purple cotton bed, a white glass window, a white Lego dresser, a white metal wall lamp, a white metal lamp, a white glass window, a blue drywall wall, and another white glass window.\n"
    # # discript2 = "Description 2: On the left, there is purple wood bed, black metal walllamp, purple and white cotton pillow, white glass window, white metal ceilingfan. On the right, there is white glass window, brown wood cart. On the forward, there is . On the rear, there is purple cotton bed, white glass window, white glass window, black metal walllamp\n"
    # discript2 = "Description 2: On the left, there is nothing. On the right, there is nothing. In front of me, there is a wood stool and a tiled floor. Behind me, there is a violet bed, a white glass window, a cream dresser, a white wall lamp, a white metal lamp, a white glass window, a blue wall, and another white glass window.\n"
    # discript2 = "Description 2: On the left, there is nothing. On the right, there is nothing. In front of me, there is a tiled floor and a wood stool. Behind me, there is a violet bed, a white glass window, a white wall lamp, a cream dresser, a white metal lamp, a blue wall, a white glass window, and another white glass window.\n"
    discript2 = "Description 2: On the left, there is pillow, computer. On the right, there is a book. In front of me, there is a wood stool. Behind me, there is a purple violet bed, a white glass window, a white wall lamp, a stool, a cream dresser, a white metal lamp, a blue wall, and a white glass window.\n"
    
    # discript1 = "Description 1: There is wallhangingdecoration, a white wood stool, a brown tile floor, a purple cotton bed, a white glass window, a white Lego dresser, a white metal wall lamp, a white metal lamp, a white glass window, a blue drywall wall, and another white glass window.\n"
    # discript2 = "Description 2: There is pillow, a book, a wood stool, a purple violet bed, a white glass window, a white wall lamp, a stool, a cream dresser, a white metal lamp, a blue wall, and a white glass window.\n"
    
    # discript1 = "Description 1: On the left, there is painting, pillow, chair. On the right, there is wallhangingdecoration, photo, bathtub. In front of me, there is clothes, pillow, bedtable, blinds. Behind me, there is curtainrail, shoes."
    # discript2 = "Description 2: On the left, there is picture, brush. On the right, there is pillow, wallhangingdecoration.In front of me, there is painting, bathtub, white cotton pillow, chair. Behind me, there is basketofsomething, shoes, clothes"
    # discript1 = "Description 1: On the left, there is glass painting, floorlamp, wallhangingdecoration, book, pillow, pillow, pillow, chair, blinds, windowframe. On the right, there is wallhangingdecoration, paper photo,photo, plastic photo, pillow,  pillow, pillow, pillow, pillow, pillow, bedtable, bedsidelamp, book, blinds, windowframe, doorframe, white towel, bathtub. In front of me, there is ceilingvent, pillow, black and blanket, wallhangingdecoration, footstool, bed,bedsidelamp, bedtable, windowframe,nblinds, windowframe, blinds, windowframe. Behind me, there is curtainrail"
    # discript2 = "Description 2: On the left, there is picture, brush, brush. On the right, there are pillow, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, bed, ceilingvent, towel. In front of me, there is painting, chandelier, pillow, pillow, curtain, clothes,  box, bag, shelf, clotheshanger, mirror, foodstand, diningtable. Behind me, there is cotton basketofsomething, shoes, clothes"
 
    # discript1 = "Description 1: You see glass painting, floorlamp, wallhangingdecoration, book, pillow, pillow, pillow, chair, blinds, windowframe,  wallhangingdecoration, photo, photo, photo, pillow,  pillow, pillow, pillow, pillow, pillow, bedtable, bedsidelamp, book, blinds, windowframe, doorframe, towel, bathtub, ceilingvent, pillow, black and blanket, wallhangingdecoration, footstool, bed,bedsidelamp, bedtable, windowframe,nblinds, windowframe, blinds, windowframe and curtainrail"
    # discript2 = "Description 2: You see picture, brush, brush, pillow, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, bed, ceilingvent, towel, painting, chandelier, pillow, pillow, curtain, clothes,  box, bag, shelf, clotheshanger, mirror, foodstand, diningtable, cotton basketofsomething, shoes, clothes"
 
    question = "These are depictions from two different vantage points. Please assess the shared objects and spatial relationship in the descriptions to determine whether these two descriptions represent the same room. Provide a response of True or False, along with supporting reasons. If you cannot decide, reply None in answer, but please aim for a conclusive response. To simplify the description, focus on larger objects."
    whole_query = role + discript1 + discript2 + question
    store_ans = []
    for i in range(20):
        chat_completion = llm.query_state_estimation(whole_query)
        store_ans.append(chat_completion)
        print(chat_completion)
    # # pdb.set_trace()

    print(store_ans)

    ### ---------------------   LOCAL EXPLOARATION TEST  -------------------------
    # Notice: add open_ai key config before test

    llm_config_path="configs/gpt_config.yaml"
    llm = GPTInterface(config_path=llm_config_path)

    goal = "sofa"
    start_question = "There is a list."
    Obs_obj_Discript = "["+ ", ".join(obs['object']) + "]"
    end_question = f"Please select one object that is most likely located near a {goal}."
    whole_query = start_question + Obs_obj_Discript + end_question

    chat_completion = llm.query_local_explore(whole_query)
    complete_response = chat_completion.choices[0].message.content.lower()
    sample_response = complete_response[complete_response.find('answer:'):]
    seperate_ans = re.split('\n|; |, | |answer:', sample_response)
    seperate_ans = [i for i in seperate_ans if i != '']
    print(seperate_ans) # ans should be separate_ans[0]
    
    llava = VLM_LLAVA()
    import time
    for i in range(10):
        start = time.time()
        output = llava.query(image, "Where is the photo taken? Please reply with one word.")
        print(output)
        end = time.time()
        print('Cost Time:', end-start)
        start = time.time()
        output = llava.query(image, "List objects in the photo with as many details as possible")
        print(output)
        end = time.time()
        print('Cost Time:', end-start)
