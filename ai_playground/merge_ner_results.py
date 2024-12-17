import numpy as np


def merge_local_ner_results(local_results, original_text):
    """
    Merge NER results into grouped entities using the original text for accurate word reconstruction.
    """
    entities = []
    current_entity = {
        "entity": local_results[0]["entity"],
        "start": local_results[0]["start"],
        "end": local_results[0]["end"],
        "scores": [local_results[0]["score"]],
    }

    for entity in local_results[1:]:
        if entity["entity"] == current_entity["entity"]:
            # Update the end position and collect scores
            current_entity["end"] = entity["end"]
            current_entity["scores"].append(entity["score"])
        else:
            # Extract word from the original text
            current_entity["word"] = original_text[
                current_entity["start"] : current_entity["end"]
            ]
            entities.append(
                {
                    "entity_group": current_entity["entity"][
                        2:
                    ],  # Strip B- or I- prefix
                    "score": float(np.mean(current_entity["scores"])),
                    "word": current_entity["word"],
                    "start": current_entity["start"],
                    "end": current_entity["end"],
                }
            )
            # Start a new entity
            current_entity = {
                "entity": entity["entity"],
                "start": entity["start"],
                "end": entity["end"],
                "scores": [entity["score"]],
            }

    # Append the last entity
    current_entity["word"] = original_text[
        current_entity["start"] : current_entity["end"]
    ]
    entities.append(
        {
            "entity_group": current_entity["entity"][2:],
            "score": float(np.mean(current_entity["scores"])),
            "word": current_entity["word"],
            "start": current_entity["start"],
            "end": current_entity["end"],
        }
    )

    return entities
