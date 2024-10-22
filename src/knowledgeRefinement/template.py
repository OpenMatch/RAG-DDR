minicpm_prompt_template = "<用户>{}<AI>"

augment_templeta = 'Background:\n{}\n\n{}'


multi_choice = 'The following is multiple choice question. Please choose the best answer choice which can answer the following question.\n{}\nAnswer:'
QA_templeta = 'Q: {}\nA:'

Mult_COT_templeta = "Please answer multiple choice question and choose the best answer choice first. Then give your explanation between [<COT] and [COT>]."
QA_COT_templeta = "Please answer the question. Then give your explanation between [<COT] and [COT>]."

COT_few_shot = 'question: {}\nAnswer:{}\n[<COT] {} [COT>]\n '
COT_question = 'question: {}\nAnswer:'


aqua_rat_shot = [{"question": "If six persons sit in a row, then the probability that three particular persons are always together is ?\\nOptions:\\n(A) 1/6\\n(B) 1/0\\n(C) 1/5\\n(D) 1/4\\n(E) 1/1", "answer": "(C)",
                  "cot":"Six persons can be arranged in a row in 6! ways. Treat the three persons to sit together as one unit then there four persons and they can be arranged in 4! ways. Again three persons can be arranged among them selves in 3! ways. Favourable outcomes = 3!4! Required probability = 3!4!/6! = 1/5"},
                 {"question": "From the set of numbers x, y, t, z, s and w, how many different combinations can we have without the t in them? Ex:. (x,y), (x), (w,z,y,x,s), etc and (x,y)=(y,x)\\nOptions:\\n(A) 10\\n(B) 14\\n(C) 15\\n(D) 16\\n(E) 31", "answer": "(E)",
                  "cot":"Another way: Any letter (x, y, z, w, s) can be included or not. So, we have 2^5 combinations - 1 empty combination = 31 combinations"},
                 {"question": "The total number of problems in the mathematics and algebra books is 28 more than the total number of problems in the algebra and geometry books. How much lesser is the number of problems in the geometry book as compared to the mathematics book?\\nOptions:\\n(A) 30\\n(B) 18\\n(C) 28\\n(D) 25\\n(E) 32","answer":"(C)",
                  "cot":"(Mathematics + Algebra) - (Algebra + Geometry) = 28\\nMathematics - Geometry = 28"}]
                  
ecqa_shot = [{"question": "The aggressive soldiers were getting drunk, what are they at they at risk of doing with the locals?\\nOptions:\\n- nausea\\n- erections\\n- fights\\n- relaxation\\n- procreation", "answer": "fights",
                 "cot": "Aggressiveness means ready or likely to attack or confront. Getting drunk encourages aggressiveness. Attacking and confronting comes under fights."},
            {"question": "What does exercising immediately lead to?\\nOptions:\\n- relaxation\\n- exhaustion\\n- energetic\\n- become stronger\\n- use energy", "answer": "exhaustion",
             "cot": "Exhaustion is caused by exercising. Exercising immediately lead to exhaustion."},
            {"question": "The blood was everywhere in the operating room, what were the surgeons operating on?\\nOptions:\\n- soccer game\\n- musle\\n- vein\\n- animals\\n- person", "answer": "person",
            "cot": "Operating room is a place. Surgeries are performed in the operating room. Surgeries are performed on human beings. Operating rooms are found in hospitals for human beings."}
            ]

gsm8k_shot = [{"question": "There are 12 carpets in house 1, 20 carpets in house 2, and 10 carpets in house 3. If house 4 has twice as many carpets as house 3, how many carpets do all 4 houses have in total?", "answer": "62",
               "cot": "House 4 has 2 * 10 house 3 carpets = 20 carpets. The total number of carpets across all houses is 12 + 20 + 10 + 20 = 62."},
              {"question": "CJ, KJ, and AJ collect stamps. CJ has 5 more than twice the number of stamps that KJ has, and KJ has half as many as AJ. If the three boys have 930 stamps all together, how many stamps does AJ have?", "answer": "370",
               "cot": "Let x represent the number of stamps for AJ. KJ:x / 2 stamps. CJ:5 + 2(x / 2) = 5 + x. Total:x + (x / 2) + 5 + x = 930. (5 / 2)x + 5 = 930. (5 / 2)x = 925. x = 925(2 / 5) = 370 stamps."},
              {"question": "Lily types 15 words a minute and takes a 2-minute break every 10 minutes. How long does it take for Lily to type 255 words?", "answer": "19",
              "cot": "It would take Lily 255 words / 15 wpm = 17 minutes to type without taking a break. Since Lily takes a break after 10 minutes of typing she takes 17 minutes + 2 minutes = 19 minutes."}]

strategyqa_shot = [{"question": "Is radioactive waste a plot device for many shows?", "answer": "yes",
                    "cot": "Radioactive isotopes in an ooze-like waste cause turtles to become the Teenage Mutant Ninja Turtles. In the Fox animated hit, Family Guy, radioactive waste is used to turn give the main characters superpowers. The superhero 'Daredevil' encounters radioactive waste that blinds him as a child and gives him super powers."},
                   {"question": "Will a celibate cleric likely suffer a stoning in Somalia?", "answer": "no",
                    "cot": "A cleric is the term for a Muslim priest. Celibate people remain chaste and do not engage in relations with others. Stoning is a penalty in Somalia used to punish adulterers. Many Islamic militants have been in control of various parts of Somalia."},
                   {"question": "Is Bill Gates the wealthiest of the Baby Boomers?", "answer": "no",
                    "cot": "The Baby Boomers are the generation born between the years 1946-1964. Bill Gates was born on October 28, 1955 and has a net worth of 108 billion as of 2020. Jeff Bezos was born on January 12, 1964 and has a net worth of 160 billion as of 2020."}
                   ]

llama_multi_choice = 'Please answer the multiple choice questions below and output only the choice.\n{}\nAnswer:'
llama_QA_COT_templeta = "Please answer the question and only output the answer. Then give your explanation between [<COT] and [COT>]."

