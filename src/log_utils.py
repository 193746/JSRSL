import jsonlines

# slurp
actions=['set', 'volume_up', 'hue_lightup', 'query', 'coffee', 'volume_mute', 'remove', 'stock', 'music', 'events', 'definition',
         'podcasts', 'addcontact', 'likeness', 'quirky', 'factoid', 'order', 'audiobook', 'cleaning', 'greet', 'taxi', 'sendemail',
         'joke', 'maths', 'post', 'ticket', 'recipe', 'settings', 'wemo_on', 'hue_lightchange', 'radio', 'querycontact', 'traffic',
         'currency', 'hue_lightoff', 'createoradd', 'locations', 'movies', 'wemo_off', 'hue_lighton', 'volume_down', 'game', 'convert',
         'hue_lightdim', 'dislikeness', 'volume_other', 'negate', 'dontcare', 'repeat', 'affirm', 'commandstop', 'confirm', 'explain',
         'praise']
scenarios=['calendar', 'audio', 'iot', 'weather', 'lists', 'email', 'alarm', 'qa', 'play', 'recommendation', 'social', 'news', 'music',
           'general', 'takeaway', 'transport', 'cooking', 'datetime']
entities=[-1, 'date', 'person', 'time', 'list_name', 'business_name', 'event_name', 'relation', 'playlist_name', 'definition_word',
         'email_address', 'timeofday', 'place_name', 'weather_descriptor', 'news_topic', 'media_type', 'food_type', 'device_type',
         'transport_type', 'joke_type', 'podcast_name', 'player_setting', 'artist_name', 'house_place', 'audiobook_name', 'radio_name',
         'music_genre', 'app_name', 'transport_name', 'color_type', 'music_descriptor', 'transport_agency', 'order_type', 'business_type',
         'change_amount', 'email_folder', 'podcast_descriptor', 'coffee_type', 'personal_info', 'currency_name', 'game_name', 'time_zone',
         'general_frequency', 'song_name', 'ingredient', 'transport_descriptor', 'movie_type', 'sport_type', 'drink_type', 'movie_name',
         'cooking_type', 'alarm_type', 'meal_type', 'music_album', 'game_type', 'audiobook_author', 'query_detail']

# aishell-ner
aishell_entities=[-1,"person","location","organization"]

# pred_text<->true_text
def write_transcription(pred_texts,true_texts,output_path):
    with open(output_path,"a",encoding="UTF-8") as f:
        for p,t in zip(pred_texts,true_texts):
            f.write(p+"<->"+t+"\n")

def write_structure_information(ner_pred_list,ner_true_list,ic_pred_list,ic_true_list,output_path,wav_dict):
    dataset_name=output_path.split("/")[-2].split("|")[1]
    target_dict={}
    if dataset_name=="AISHELL-NER":
        for item in ner_pred_list:
            if str(item[0]) in target_dict.keys():
                target_dict[str(item[0])]["pred_entities"].append({"type":aishell_entities[int(item[1])],"filler":"".join(item[2])})
            else:
                target_dict[str(item[0])]={
                    "file":wav_dict[str(item[0])],
                    "pred_entities":[{"type":aishell_entities[int(item[1])],"filler":"".join(item[2])}],
                    "true_entities":[]}
        for item in ner_true_list:
            if str(item[0]) in target_dict.keys():
                target_dict[str(item[0])]["true_entities"].append({"type":aishell_entities[int(item[1])],"filler":"".join(item[2])})
            else:
                target_dict[str(item[0])]={
                    "file":wav_dict[str(item[0])],
                    "pred_entities":[],
                    "true_entities":[{"type":aishell_entities[int(item[1])],"filler":"".join(item[2])}]}
    elif dataset_name=="SLURP":
        for item in ic_pred_list:
            target_dict[str(item[0])]={
                    "file":wav_dict[str(item[0])],
                    "entities":[],
                    "action":actions[item[1]],
                    "scenario":scenarios[item[2]]}
        for item in ner_pred_list:
            target_dict[str(item[0])]["entities"].append({"type":entities[item[1]],"filler":" ".join(item[2])})
    else:
        raise Exception(f"unknown dataset:{args.dataset_name}")

    with jsonlines.open(output_path, mode='a') as writer:
        for item in target_dict.values():
            writer.write(item)