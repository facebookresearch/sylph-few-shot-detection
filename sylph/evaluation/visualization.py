from d2go.utils.visualization import DataLoaderVisWrapper
from detectron2.utils.events import get_event_storage


class EpisodicLearningDataLoaderVisWrapper(DataLoaderVisWrapper):
    """
    Wrap the data loader to visualize its output via TensorBoardX at given frequency.
    """

    def __init__(self, cfg, tbx_writer, data_loader):
        super().__init__(cfg, tbx_writer, data_loader)

    def _maybe_write_vis(self, data, name):
        try:
            storage = get_event_storage()
        except AssertionError:
            # wrapped data loader might be used outside EventStorage, don't visualize
            # anything
            return

        if (
            self.log_frequency == 0
            or not storage.iter % self.log_frequency == 0
            or self._remaining <= 0
        ):
            return

        length = min(len(data), self._remaining)
        data = data[:length]
        self._remaining -= length

        for i, per_image in enumerate(data):
            vis_image = self._visualizer.visualize_train_input(per_image)
            tag = f"{name}_{storage.iter}"
            if "dataset_name" in per_image:
                tag += per_image["dataset_name"] + "/"
            if "file_name" in per_image:
                tag += "img_{}/{}".format(i, per_image["file_name"])
            self.tbx_writer._writer.add_image(
                tag=tag,
                img_tensor=vis_image,
                global_step=storage.iter,
                dataformats="HWC",
            )

    def __iter__(self):
        for data in self.data_loader:
            shot = len(data[0]["support_set"])
            # only show the first example of the support set
            batched_inputs_support_set = [x["support_set"][0] for x in data]  # 1* (c*s)
            name = f"train_loader_support_set_batch_{shot}_shot"
            self._maybe_write_vis(batched_inputs_support_set, name)

            shot = len(data[0]["query_set"])
            batched_inputs_query_set = [
                record for x in data for record in x["query_set"]
            ]  # 1* (c*s)
            name = f"train_loader_query_set_batch_{shot}_shot"

            self._maybe_write_vis(batched_inputs_query_set, name)
            yield data
