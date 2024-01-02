import openai
# import dotenv
import pickle
import retry
import os
import numpy as np

# dotenv.load_dotenv(".env", override=True)

def set_openai_key(key, org):
    """Sets OpenAI key."""
    openai.api_key = key
    if org != None:
        openai.organization = org
    print("Key successfully set.")
    print("Org:", org)

set_openai_key(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG"))

test_set = """You see a room with a leash, pet bowls, and a scratching post.
You find a litter box.

You see a room with wrapping paper, ribbons, and greeting cards.
You find a tape dispenser.

You see a room with cookie sheets, mixing bowls, and a stand mixer.
You find a muffin tin.

You see a room with laundry detergent, fabric softener, and a clothes hamper.
You find a clothes dryer.

You see a room with a toothbrush, a towel, and a bar of soap.
You find a shower curtain.

You see a room with a toolbox, hammer, and a drill.
You find a screwdriver set.

You see a room with a bird feeder, gardening gloves, and a potted plant.
You find a watering can."""
test_trajs = test_set.split("\n")[::3]
test_goals = test_set.split("\n")[1::3]

@retry.retry(tries=3, delay=1)
def get_completion(prompt, max_tokens=100, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0, engine="davinci", echo=True):

    # Save the prompt to a text file in tmp, give the file a random name
    # with open(f"tmp/prompt_{random.random()}.txt", "w+") as f:
    #      f.write(str(prompt))

    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=5,
        echo=echo)
    return response


PREFACE = """You are a robot navigating through a house looking for various objects. You are given a sequence of observations and should list what the robot is likely to find next.

"""

PREFACE_WITH_PROMPTING = """You are a robot navigating through a house looking for various objects. You are given a sequence of observations and should list what the robot is likely to find next.

You see a room with a table, microwave, and a chair.
You find a refrigerator.

You see a room with a table, microwave, and a chair.
You find a refrigerator.

You see a room with a desk, a laptop, and a bookshelf.
You find a printer.

You see a room with a bathtub, sink, and a towel rack.
You find a toilet.

You see a room with a dining table, chairs, and a china cabinet.
You find a kitchen island.

You see a room with a treadmill, dumbbells, and an exercise mat.
You find an elliptical machine.

You see a room with a playpen, stuffed animals, and a changing table.
You find a crib.

You see a room with a lawnmower, rake, and a wheelbarrow.
You find a garden hose.

You see a room with a pool table, dart board, and a bar.
You find a foosball table.

You see a room with a washing machine, clothes hamper, and an ironing board.
You find a dryer.

You see a room with a bed, a bedside table, and a wardrobe.
You find a vanity.

You see a room with a couch, a coffee table, and a TV.
You find a sound system.

"""

def score_plan(goal, trajectory):

    goal_prompt = f"You find {goal}"
    if type(trajectory) == list:
        prompt = [PREFACE + x + "\n" + goal_prompt for x in trajectory]
    else:
        prompt = PREFACE + trajectory + "\n" + goal_prompt

    # for p in prompt:
    #     print(p, "\n")
    res = get_completion(prompt, max_tokens=0, temperature=0, engine="davinci")
        
    def parse_probs(choice, prompt):
        tokens = choice["logprobs"]["tokens"]
        logprobs = choice["logprobs"]["token_logprobs"]
        offsets = choice['logprobs']["text_offset"]

        # Find first offset that is greater than len(trajectory)
        for i, offset in enumerate(offsets):
            if offset > len(prompt) - len(goal):
                break
        # Get the logprobs for the goal prompt
        # goal_tokens = tokens[i-1:]
        goal_logprobs = logprobs[i-1:]
        return sum(goal_logprobs)

    if type(trajectory) == list:
        return [parse_probs(x, prompt[i]) for i, x in enumerate(res["choices"])]
    return parse_probs(res["choices"][0], prompt)

def object_query_constructor(objects):
    """
    Construct a query string based on a list of objects

    Args:
        objects: torch.tensor of object indices contained in a room

    Returns:
        str query describing the room, eg "This is a room containing
            toilets and sinks."
    """
    assert len(objects) > 0
    # query_str = "This room contains "
    query_str = "You see "
    names = []
    for ob in objects:
        names.append(ob)
    if len(names) == 1:
        query_str += names[0]
    elif len(names) == 2:
        query_str += names[0] + " and " + names[1]
    else:
        for name in names[:-1]:
            query_str += name + ", "
        query_str += "and " + names[-1]
    query_str += "."
    return query_str



USER_EXAMPLE_1 = """You see the following clusters of objects:

1. door
2. sofa, plant
3. bed, plant, table

Question: Your goal is to find a toilet. Where should you go next?
"""

AGENT_EXAMPLE_1 = """Reasoning: a bathroom is usually attached to a bedroom so it is likely that if you explore a bedroom you will find a bathroom and thus find a toilet
Answer: 3
"""

USER_EXAMPLE_2 = """You see the following clusters of objects:

1. plant
2. bed, chair, dresser

Question: Your goal is to find a tv. Where should you go next?
"""

AGENT_EXAMPLE_2 = """Reasoning: The tv is not likely to be in a bedroom but a plant does not provide enough information.
Answer: 0
"""

import re
def find_first_integer(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        raise ValueError('No integer found in string')

@retry.retry(tries=5)
def ask_gpt(goal, object_clusters):
    system_message = "You are a robot exploring a house. You have access to semantic sensors that can detect objects. You are in the middle of the house with clusters of objects. Your goal is to figure near which cluster to explore next. Always provide reasoning and if there is no clear choice select answer 0" 
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": USER_EXAMPLE_1},
        {"role": "assistant", "content": AGENT_EXAMPLE_1},
        {"role": "user", "content": USER_EXAMPLE_2},
        {"role": "assistant", "content": AGENT_EXAMPLE_2}
    ]
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string}\n"
        messages.append({"role": "user", "content": f"You see the following clusters of objects:\n\n {options}\nQuestion: You goal is to find a {goal}. Where should you go next? If there is not clear choice select answer 0.\n"})
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages)
        complete_response = completion.choices[0].message.content
        # Make the response all lowercase
        complete_response = complete_response.lower()
        reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
        # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
        answer = int(find_first_integer(complete_response.split("answer")[1]))
        return answer, reasoning
    raise Exception("Object categories must be non-empty")

@retry.retry(tries=5)
def ask_gpts(goal, object_clusters, num_samples=10):
    system_message = "You are a robot exploring a house. You have access to semantic sensors that can detect objects. You are in the middle of the house with clusters of objects. Your goal is to figure near which cluster to explore next. Always provide reasoning and if there is no clear choice select answer 0" 
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": USER_EXAMPLE_1},
        {"role": "assistant", "content": AGENT_EXAMPLE_1},
        {"role": "user", "content": USER_EXAMPLE_2},
        {"role": "assistant", "content": AGENT_EXAMPLE_2}
    ]
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string}\n"
        messages.append({"role": "user", "content": f"You see the following clusters of objects:\n\n {options}\nQuestion: You goal is to find a {goal}. Where should you go next? If there is not clear choice select answer 0.\n"})
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo", temperature=1,
            n=num_samples, messages=messages)
        
        answers = []
        reasonings = []
        for choice in completion.choices:
            try:
                complete_response = choice.message.content
                # Make the response all lowercase
                complete_response = complete_response.lower()
                reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                answer = int(find_first_integer(complete_response.split("answer")[1]))
                answers.append(answer)
                reasonings.append(reasoning)
            except:
                continue

        unique_answers = list(set(answers))
        # It is possible GPT gives an invalid answer less than zero or greater than 1 plus the number of object clusters. Remove invalid answers
        unique_answers = [x for x in unique_answers if x >= 0 and x <= len(object_clusters)]
        answers = [x for x in answers if x >= 0 and x <= len(object_clusters)]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {x: answers.count(x) / len(answers) for x in unique_answers}
        return answer_counts, reasonings
    raise Exception("Object categories must be non-empty")


V2_SYSTEM_PROMPT_NEGATIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide reasoning along with a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Reasoning: Knifes are typically not kept in a living room or office space which is what the objects in 1 and 2 suggest. Therefore you should avoid looking in 1 and 2.
Answer: 1,2


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgement when determining what room a cluster of objects is likely to be in.
"""

V2_SYSTEM_PROMPT_POSITIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Assistant:
Reasoning: Knifes are typically kept in the kitchen and a sink, microwave, and refrigerator are commonly found in kitchens. Therefore we should check the cluster that is likely to be a kitchen first.
Answer: 3


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgement when determining what room a cluster of objects is likely to be in.
"""

V2_SYSTEM_PROMPT_NEGATIVE_NO_REASONING = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Answer: 1,2


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgement when determining what room a cluster of objects is likely to be in.
"""

V2_SYSTEM_PROMPT_POSITIVE_NO_REASONING = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Answer: 3


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgement when determining what room a cluster of objects is likely to be in.
"""

cache_filename = 'gpts_v2_cache.pkl'

# Load the cache from a file if it exists
if os.path.exists(cache_filename):
    with open(cache_filename, 'rb') as f:
        cache = pickle.load(f)
else:
    cache = {}

@retry.retry(tries=5)
def ask_gpts_v2(goal, object_clusters, env="a house", positives=True, num_samples=10, model="gpt-3.5-turbo", reasoning_enabled=True):
    global cache

    # Handle caching of the results
    key = (goal, tuple(object_clusters), env, positives, num_samples, model, reasoning_enabled)
    if key in cache:
        return cache[key]

    if reasoning_enabled:
        if positives:
            system_message = V2_SYSTEM_PROMPT_POSITIVE
        else:
            system_message = V2_SYSTEM_PROMPT_NEGATIVE
    else:
        if positives:
            system_message = V2_SYSTEM_PROMPT_POSITIVE_NO_REASONING
        else:
            system_message = V2_SYSTEM_PROMPT_NEGATIVE_NO_REASONING


    messages=[
        {"role": "system", "content": system_message},
    ]
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string[:-2]}\n"
        if positives:
            messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?"})
        else:
            messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I avoid spending time searching if I am looking for {goal}?"})
        
        completion = openai.chat.completions.create(
            model=model, temperature=1,
            n=num_samples, messages=messages)
        
        answers = []
        reasonings = []

        for choice in completion.choices:
            try:
                complete_response = choice.message.content
                # Make the response all lowercase
                complete_response = complete_response.lower()
                if reasoning_enabled:
                    reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                else:
                    reasoning = "disabled"
                # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                answer = complete_response.split("answer: ")[1].split("\n")[0]
                # Separate the answers by commas
                answers.append([int(x) for x in answer.split(",")])
                reasonings.append(reasoning)
            except:
                answers.append([])

        # Flatten answers
        flattened_answers = [item for sublist in answers for item in sublist]
        # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
        filtered_flattened_answers = [x for x in flattened_answers if x >= 1 and x <= len(object_clusters)]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {x: filtered_flattened_answers.count(x) / len(answers) for x in set(filtered_flattened_answers)}
        
        # Handle caching of the results
        # Load the most recent cache from a file if it exists
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                cache = pickle.load(f)
        cache[key] = (answer_counts, reasonings)
        with open(cache_filename, 'wb') as f:
            pickle.dump(cache, f)

        return answer_counts, reasonings
    raise Exception("Object categories must be non-empty")

EXAMPLE = """You are a robot exploring a house. You have access to semantic sensors that can detect objects. You are in the middle of the house with clusters of objects. Your goal is to figure near which cluster to explore next. If there is not clear choice select answer 0.

You see the following clusters of objects:

0. no clear option
1. shower
2. sofa, plant, TV
3. bed, plant, table

Question: You goal is to find a toilet. Where should you go next? 
Answer: 1"""

def score_clusters(goal, object_clusters):
    preface = "\n\nYou see the following clusters of objects:\n\n 0. no clear option\n"
    options = ""
    for i, cluster in enumerate(object_clusters):
        cluser_string = ""
        for ob in cluster:
            cluser_string += ob + ", "
        options += f"{i+1}. {cluser_string}\n"
    question = f"Question: Your goal is to find a {goal}. Where should you go next?\nAnswer:"
    prompt = EXAMPLE + preface + options + "\n" + question
    res = get_completion(prompt, max_tokens=1, temperature=0, engine="davinci", echo=False)
    choice = res["choices"][0]
    logprob = choice["logprobs"]["token_logprobs"][0]

    # Compute the probability by getting the logprob scores for each other option 0, 1, 2, 3, ... and normalizing
    top_logprobs = choice["logprobs"]["top_logprobs"][0]
    total_probability_mass = 0
    for key, value in top_logprobs.items():
        # If we can cast the key to an int, do it and check the value
        try:
            key = int(key)
            if 0 <= key < len(object_clusters):
                total_probability_mass += np.exp(value)
        except:
            continue
    print("total_probability_mass", total_probability_mass)
    print("logprob", logprob)
    prob = np.exp(logprob) / total_probability_mass
    return int(choice["text"]), logprob, prob


def query_llm(method: int, object_clusters: list, goal: str, save_reasoning: bool = False, reasoning_file: str = "", timestep: int = 0, reasoning_enabled: bool = True) -> list:
    """
    Query the LLM fore a score and a selected goal. Returns a list of language scores for each target point
    method = 0 uses the naive single sample LLM and binary scores of 0 or 1
    method = 1 uses the sampling based approach and gives scores between 0 and 1
    """

    # Convert object clusters to a tuple of tuples so we can hash it and get unique elements
    object_clusters_tuple = [tuple(x) for x in object_clusters]
    # Remove empty clusters and duplicate clusters
    query = list(set(tuple(object_clusters_tuple)) - set({tuple([])}))

    if method == 0:
            try:
                goal_id, reasoning = ask_gpt(goal, query)
            except Exception as excptn:
                goal_id, reasoning = 0, "GPT failed"
                print("GPT failed:", excptn)
            if goal_id != 0:
                goal_id = np.argmax([1 if x == query[goal_id - 1] else 0 for x in object_clusters_tuple]) + 1
            language_scores = [0] * (len(object_clusters_tuple) + 1)
            language_scores[goal_id] = 1
    elif method == 1:
        try: 
            answer_counts, reasoning = ask_gpts(goal, query)
        except Exception as excptn:
            answer_counts, reasoning = {}, "GPTs failed"
            print("GPTs failed:", excptn)
        language_scores = [0] * (len(object_clusters_tuple) + 1)
        for key, value in answer_counts.items():
            if key != 0:
                for i, x in enumerate(object_clusters_tuple):
                    if x == query[key - 1]:
                        language_scores[i + 1] = value
            else:
                language_scores[0] = value
    elif method == 2:
        # try:
        answer_counts, reasoning = ask_gpts_v2(goal, query, positives=True, reasoning_enabled=reasoning_enabled)
        # except Exception as excptn:
        #     answer_counts, reasoning = {}, "GPTs failed"
        #     print("GPTs failed:", excptn)
        language_scores = [0] * len(object_clusters_tuple)
        for key, value in answer_counts.items():
            for i, x in enumerate(object_clusters_tuple):
                if x == query[key - 1]:
                    language_scores[i] = value

                # Save reasoning to a file
        if save_reasoning:
            with open(reasoning_file, "a") as f:
                f.write(f"Timestep: {timestep}\n")
                f.write(f"Goal: {goal}\n")
                f.write(f"Query: {query}\n")
                f.write(f"Answer counts: {answer_counts}\n")
                f.write(f"Reasoning: {reasoning}\n")
                f.write(f"Object clusters: {object_clusters}\n")
                f.write(f"Language scores: {language_scores}\n\n")

    elif method == 3:
        try:
            answer_counts, reasoning = ask_gpts_v2(goal, query, positives=False)
        except Exception as excptn:
            answer_counts, reasoning = {}, "GPTs failed"
            print("GPTs failed:", excptn)
        language_scores = [0] * len(object_clusters_tuple)
        for key, value in answer_counts.items():
            for i, x in enumerate(object_clusters_tuple):
                if x == query[key - 1]:
                    language_scores[i] = value
            
        # Save reasoning to a file
        if save_reasoning:
            with open(reasoning_file, "a") as f:
                f.write(f"Timestep: {timestep}\n")
                f.write(f"Goal: {goal}\n")
                f.write(f"Query: {query}\n")
                f.write(f"Answer counts: {answer_counts}\n")
                f.write(f"Reasoning: {reasoning}\n")
                f.write(f"Object clusters: {object_clusters}\n")
                f.write(f"Language scores: {language_scores}\n\n")

    # elif method == 2 or method == 3:
    #     # Use the L3MVN finetune baseline
    #     if method == 2:
    #         reasoning = "L3MVN finetune"
    #         scores = send_request(5000, goal, query)
    #         # Set any language scores that are less than 0.2 to 0 (done in paper)
    #         scores = [0 if x < 0.2 else x for x in scores]
    #     elif method == 3:
    #         reasoning = "L3MVN zeroshot"
    #         scores = send_request(5001, goal, query)
    #         # Set any language scores that are less than 0.3 to 0 (done in paper)
    #         scores = [0 if x < 0.3 else x for x in scores]
    #     # Map back to the original object clusters
    #     language_scores = [0] * (len(object_clusters) + 1)
    #     # Align query back with object clusters
    #     for x in object_clusters_tuple:
    #         for i, y in enumerate(query):
    #             if x == y:
    #                 language_scores[object_clusters_tuple.index(x) + 1] = scores[i]
    else:
        raise Exception("Invalid method")
    
    # The first element of language scores is the scores for uncertain, the last n-1 correspond to the semantic scores for each point
    return language_scores, reasoning

def aggregate_reasoning(reasoning: list):
    # Pass in a list of reasoning strings and aggregate them into a single string
    # Ask GPT to aggregate the reasoning into a single consensus

    # Construct the prompt
    system_prompt = "You are given a series of explanations regarding where to navigate in order to find an object. You should aggregate the reasoning from multiple agents into a single sentence"
    prompt = ""
    for i, r in enumerate(reasoning):
        prompt += f"Reasoning {i}: {r}\n"

    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages)
    complete_response = completion.choices[0].message.content
    return complete_response