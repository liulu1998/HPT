import json
from to_pt import id2label, label2id


FILES_TO_PREPROCESS = [
    "./yixinli_raw_train.json",
    "./yixinli_raw_val.json",
    "./yixinli_raw_test.json",
]

TARGET_FILES = [
    "./yixinli_train.json",
    "./yixinli_dev.json",
    "./yixinli_test.json",
]


def parse_data(input_file_path, output_file_path):
    def process_one_sample(s: dict) -> dict:
        labels = s["label"]
        label_ids = list()
        for label in labels:
            label_ids.append(label2id[label])

        assert len(label_ids) == len(labels), "number of label does not match !"
        s["label"] = label_ids
        s["token"] = s["text"]
        del s["brief"]
        del s["text"]
        return s
    # <<< inner method

    samples = list()
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            samples.append(process_one_sample(data))

    with open(output_file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")


if __name__ == '__main__':
    for in_f, out_f in zip(FILES_TO_PREPROCESS, TARGET_FILES):
        parse_data(in_f, out_f)
