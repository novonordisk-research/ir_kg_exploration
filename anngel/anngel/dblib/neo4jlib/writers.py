from typing import TYPE_CHECKING, Any, List
from contextlib import contextmanager
import csv

if TYPE_CHECKING:
    from neo4j import Result


def write_to_pandas_dataframe(output_file: Any, result: "Result"):
    import pandas as pd

    record = next(result)
    columns = record.keys()
    records = [record.values()]

    for record in result:
        records.append(record.values())

    return pd.DataFrame.from_records(data=records, columns=columns)


@contextmanager
def csv_writer(file, mode="w", **kwargs):
    csvfile = open(file=file, mode=mode, encoding="utf-8")
    writer = csv.writer(csvfile, lineterminator="\n", **kwargs)
    try:
        yield writer
    finally:
        csvfile.close()


def write_records_to_csv(output_file: str, result: "Result"):
    from pathlib import Path

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with csv_writer(output_file, mode="w") as writer:
        record = next(result)
        writer.writerow(record.keys())
        writer.writerow(record.values())
        for record in result:
            writer.writerow(record.values())

    return output_file


def write_features_to_csv(output_file: str, result: "Result", properties: List[str]):
    from pathlib import Path

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def append_record(writer, record, properties):
        d = {k: None for k in properties}
        d["id"] = int(record.values()[0].element_id)
        d.update(record.value().items())

        writer.writerow(d.values())

    def write_header(writer, properties):
        writer.writerow(properties)

    if not output_file.endswith(".csv"):
        output_file += ".csv"
    with csv_writer(output_file, mode="w") as writer:
        write_header(writer=writer, properties=properties)
        for record in result:
            append_record(writer=writer, record=record, properties=properties)

    return output_file
