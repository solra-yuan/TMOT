import copy
import logging

import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from torchvision.ops.boxes import clip_boxes_to_image
from visdom import Visdom
from packaging.version import Version

from .util.plot_utils import fig_to_numpy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .NormalizeHelper import NormalizeHelper

logging.getLogger('visdom').setLevel(logging.CRITICAL)


def get_hsv_color_map(lutsize: int):
    """
    Retrieve an HSV colormap adjusted to the specified lutsize.

    If the version of matplotlib is 3.8.0 or higher, it uses the new colormaps interface 
    to resample the 'hsv' colormap according to lutsize. Otherwise, it resorts to 
    plt.cm.get_cmap to obtain a colormap adjusted to the specified lutsize.

    Parameters:
    lutsize (int): The size to resample the colormap.

    Returns:
    Colormap: The resampled 'hsv' colormap.
    """

    # Check the version of matplotlib to use the appropriate method for fetching the colormap.
    if Version(matplotlib.__version__) >= Version("3.8.0"):
        # For matplotlib 3.8.0 or newer, use the colormaps interface.
        hsv = matplotlib.colormaps['hsv'].resampled(lutsize)
    else:
        # Otherwise, use plt.cm.get_cmap.
        hsv = plt.cm.get_cmap('hsv', lutsize)

    return hsv


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None,
                 resume=False, port=8097, server='http://localhost'):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'
        self.viz = Visdom(env=env, port=port, server=server)
        # if resume first plot should not update with replace
        self.removed = not resume

    def win_exists(self):
        return self.viz.win_exists(self.win)

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None

    def register_event_handler(self, handler):
        self.viz.register_event_handler(handler, self.win)


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update = self.update_mode
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update = None
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(
            X=X,
            Y=Y,
            opts=self.viz_opts,
            win=self.win,
            update=update
        )

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        # TODO: currently reset does not empty directly only on the next plot.
        # update='remove' is not working as expected.
        if self.win is not None:
            # self.viz.line(X=None, Y=None, win=self.win, update='remove')
            self.removed = True


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        self.win = self.viz.images(
            images,
            nrow=1,
            opts=self.viz_opts,
            win=self.win,
        )
        self.viz.save([self.viz.env])


def draw_text(ax, x, y, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5)):
    ax.text(x, y, text, fontsize=fontsize, bbox=bbox)


def draw_track_id(ax, x, y, track_id, fontsize=10, bbox=dict(facecolor='white', alpha=0.5)):
    """Displays the tracking ID on the graphic."""
    draw_text(ax, x, y, f"track_id={track_id}", fontsize=fontsize,  bbox=bbox)


def draw_label(ax, x, y, lable, fontsize=10, bbox=dict(facecolor='white', alpha=0.5)):
    """Displays the tracking ID on the graphic."""
    draw_text(ax, x, y+10,  f"lable={lable}", fontsize=fontsize, bbox=bbox)


def draw_rectangle(ax, x1, y1, x2, y2, fill=False, color='green', linewidth=2):
    """Draws a rectangle on the graphic."""
    ax.add_patch(plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=fill,
        color=color,
        linewidth=linewidth
    ))


def draw_mask(ax, mask, cmap, alpha=0.5):
    """
    Displays the mask on the graphic.

    Parameters:
    - ax: The matplotlib axes to draw on.
    - mask: The mask data to visualize.
    - cmap: The colormap instance or a color string to use for the mask.
    - alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
    """

    mask_is_zero = np.isclose(mask, 0.0, atol=1e-8)
    masked_mask = np.ma.masked_where(mask_is_zero, mask)
    ax.imshow(masked_mask, alpha=alpha, cmap=cmap)


def vis_previous_frames(ax, frame_target, get_cmap):
    """
    Visualizes previous frames with tracking IDs, bounding boxes, and masks.

    Parameters:
    - ax: The matplotlib axes to draw on.
    - frame_target: A dictionary containing 'track_ids', 'boxes', and optionally 'masks'.
    - get_cmap: A function that takes an index and returns a colormap.
    """
    for j, track_id in enumerate(frame_target['track_ids']):
        x1, y1, x2, y2 = frame_target['boxes'][j]
        draw_text(
            ax,
            x1,
            y1,
            f"track_id: {track_id}\nlable: {frame_target['labels'][j]}"
        )
        draw_rectangle(ax, x1, y1, x2, y2)

        if 'masks' in frame_target:
            mask = frame_target['masks'][j].cpu().numpy()
            # Assuming get_cmap returns a colormap for the current index.
            cmap = get_cmap(j)
            draw_mask(ax, mask, cmap)


def append_legend_handles_for_unmatched_track_queries(
    track_queries_fal_pos_mask,
    num_track_queries,
    num_track_queries_with_id,
    keep,
    legend_handles
):
    """
    Appends legend handles for track queries without matching IDs to the legend.
    This function targets cases where the number of track queries with IDs does not match
    the total number of track queries, indicating the presence of false or unmatched track queries.

    Parameters:
    - legend_handles: List of existing legend handles to append to.
    - keep: Boolean mask for detections considered for visualization.
    - track_queries_fal_pos_mask: Mask indicating false positive track queries.
    - num_track_queries: Total number of track queries.
    - num_track_queries_with_id: Number of track queries with an associated ID.
    """
    if num_track_queries_with_id == num_track_queries:
        return

    fal_pos_track_queries_count = keep[track_queries_fal_pos_mask].sum()
    fal_pos_label = f"Unmatched track queries ({fal_pos_track_queries_count}/{num_track_queries - num_track_queries_with_id})"
    legend_handles.append(mpatches.Patch(color='red', label=fal_pos_label))


def append_track_queries_legend_if_exists(
    num_track_queries: int,
    num_track_queries_with_id: int,
    track_queries_mask,
    track_queries_fal_pos_mask,
    keep: torch.Tensor,
    legend_handles: list
):
    """
    Appends a legend for track queries if they exist.

    Args:
        num_track_queries (int): The number of track queries.
        num_track_queries_with_id (int): The number of track queries with an ID.
        target (dict): The target object containing tracking information.
        keep (torch.Tensor): A tensor indicating which detections to keep based on score comparison.
        legend_handles (list): The list of legend handles to append new legend entries.

    Returns:
        None: Modifies legend_handles in-place by appending new legend entry if track queries exist.
    """
    if not num_track_queries:
        return

    # Compute the number of track queries that are not false positives and format the label.
    track_queries_label = (
        f"Track queries ({keep[track_queries_mask].sum() - keep[track_queries_fal_pos_mask].sum()}"
        f"/{num_track_queries_with_id})\n- Track ID\n- Classification score\n- IoU")

    # Append the formatted label as a new legend entry.
    legend_handles.append(mpatches.Patch(
        color='blue',
        label=track_queries_label))


def create_legend_handles(
        target: dict,
        keep: torch.Tensor,
        query_keep: torch.Tensor,
        num_track_queries: int,
        num_track_queries_with_id: int
) -> list:
    """
    Creates legend handles based on the detection results and tracking information.

    Args:
        target (dict): The target object containing tracking information.
        keep (torch.Tensor): A tensor indicating which detections to keep based on score comparison.
        num_track_queries (int): The number of track queries.
        num_track_queries_with_id (int): The number of track queries with an ID.

    Returns:
        list: A list of legend handles.
    """
    legend_handles = [
        mpatches.Patch(
            color='green',
            label=f"Object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id})\n- Classification score"
        )
    ]

    # Convert masks to the appropriate device.
    track_queries_mask = target['track_queries_mask'].to(keep.device)
    track_queries_fal_pos_mask = target['track_queries_fal_pos_mask'].to(
        keep.device)

    # Append track queries related legend entries if they exist.
    append_track_queries_legend_if_exists(
        num_track_queries,
        num_track_queries_with_id,
        track_queries_mask,
        track_queries_fal_pos_mask,
        keep,
        legend_handles
    )

    # Append legend entries for unmatched track queries if any.
    append_legend_handles_for_unmatched_track_queries(
        track_queries_fal_pos_mask,
        num_track_queries,
        num_track_queries_with_id,
        keep,
        legend_handles
    )

    return legend_handles


def visualize_frame_targets(axarr, target, frame_prefixes=['prev', 'prev_prev']):
    """
    Visualizes the tracking information for previous frames.

    Parameters:
    - axarr: The array of axes objects to draw the visualizations on.
    - target: The target dictionary containing tracking information.
    - frame_prefixes: A list of frame prefixes to visualize.
    """
    # Initialize the index for subplot axes.
    i = 0

    for frame_prefix in frame_prefixes:
        # Check if the target contains the target information for the current frame prefix.
        if f'{frame_prefix}_target' not in target:
            continue

        frame_target = target[f'{frame_prefix}_target']
        num_track_ids = len(frame_target['track_ids'])

        get_cmap = get_hsv_color_map(num_track_ids)

        vis_previous_frames(axarr[i], frame_target, get_cmap)
        i += 1


def prepare_images_and_ids(target, img, frame_prefixes, inv_normalize):
    img_groups = [[inv_normalize(img[:3]).cpu(), img[3:].cpu()]]
    img_ids = [target['image_id'].item()]

    for key in frame_prefixes:
        if f'{key}_image' in target:
            img_groups.append(
                [inv_normalize(target[f'{key}_image'][:3]).cpu(), img[3:].cpu()])
            img_ids.append(target[f'{key}_target'][f'image_id'].item())

    return img_groups, img_ids


def setup_figure(img_groups, dpi=96):
    figure, axarr = plt.subplots(len(img_groups))
    figure.tight_layout()
    figure.set_dpi(dpi)
    figure.set_size_inches(
        img_groups[0][0].shape[2] * 2 / dpi,
        img_groups[0][0].shape[1] * len(img_groups) / dpi
    )

    if len(img_groups) == 1:
        axarr = [axarr]

    return figure, axarr


def display_images(axarr, img_groups, img_ids):
    axs = []
    for ax, img_group, img_id in zip(axarr, img_groups, img_ids):
        ax.set_axis_off()

        ax1 = inset_axes(
            ax,
            width="50%",
            height="100%",
            loc='center left',
            borderpad=0
        )
        ax1.imshow(img_group[0].permute(1, 2, 0).clamp(0, 1))
        ax1.set_axis_off()  # 축 제거

        ax2 = inset_axes(
            ax,
            width="50%",
            height="100%",
            loc='center right',
            borderpad=0
        )
        ax2.imshow(img_group[1].squeeze(), cmap='gray')
        ax2.set_axis_off()  # 축 제거

        draw_text(ax1, 0, 0, f'IMG_ID={img_id}')

        axs.append(ax1)
        axs.append(ax2)

    return axs


def process_and_visualize_box(
    ax,
    box_id,
    prop_i,
    keep,
    result,
    target,
    tracking,
    track_ids,
    cmap
):
    """
    Processes and visualizes a single bounding box based on tracking and keep conditions.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the visuals on.
        box_id (int): The index of the current bounding box.
        prop_i
        keep (numpy.ndarray): An array indicating which boxes to keep.
        result (dict): The result dictionary containing 'scores', 'boxes', and optionally 'masks'.
        target (dict): The target dictionary containing tracking information.
        tracking (bool): Indicates whether tracking is enabled.
        track_ids (list): List of track IDs corresponding to each box.
        cmap (matplotlib.colors.ListedColormap): The colormap used for masks visualization.

    Returns:
        None
    """
    rect_color = 'red' if tracking and target['track_queries_fal_pos_mask'][box_id] else 'green'
    offset = 50 if tracking and target['track_queries_mask'][box_id] else 0
    class_id = result['labels'][box_id]
    text = f"class: {class_id}\n" + \
        f"score(one class): {result['scores'][box_id]:0.2f}"

    if tracking:
        if target['track_queries_fal_pos_mask'][box_id]:
            rect_color = 'red'
        elif target['track_queries_mask'][box_id]:
            rect_color = 'blue'
            text = f"class: {class_id}({result['class_scores'][box_id][class_id]:0.2f})\n" + \
                f"track: {track_ids[prop_i]}({result['track_queries_with_id_iou'][prop_i]:0.2f})"
            prop_i += 1

    if not keep[box_id]:
        return prop_i

    result_boxes = clip_boxes_to_image(result['boxes'], target['size'])
    x1, y1, x2, y2 = result_boxes[box_id]

    draw_rectangle(ax, x1, y1, x2, y2, color=rect_color)
    draw_text(ax, x1, y1 + offset, text)

    if 'masks' in result:
        mask = result['masks'][box_id][0].numpy()
        draw_mask(ax, mask, cmap=colors.ListedColormap([cmap(box_id)]))

    return prop_i


def process_and_visualize_boxes(
    ax,
    keep,
    result,
    target,
    tracking,
    track_ids,
    cmap
):
    """
    Iterates over each box and visualizes it based on the keep condition and tracking information.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the visuals on.
        keep (numpy.ndarray): An array indicating which boxes to keep.
        result (dict): The result dictionary containing 'scores', 'boxes', and optionally 'masks'.
        target (dict): The target dictionary containing tracking information.
        tracking (bool): Indicates whether tracking is enabled.
        track_ids (list): List of track IDs corresponding to each box.
        cmap (matplotlib.colors.ListedColormap): The colormap used for masks visualization.

    Returns:
        None
    """

    # Counter for the property index, used when tracking is enabled
    prop_i = 0

    for box_id in range(len(keep)):
        prop_i = process_and_visualize_box(
            ax,
            box_id,
            prop_i,
            keep,
            result,
            target,
            tracking,
            track_ids,
            cmap
        )

def vis_results(
    visualizer,
    img,
    result,
    target,
    tracking,
    features
):
    frame_prefixes = ['prev', 'prev_prev']

    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    img_groups, img_ids = prepare_images_and_ids(
        target,
        img,
        frame_prefixes,
        inv_normalize
    )

    # img.shape=[3, H, W]
    figure, axarr = setup_figure(img_groups)
    num_track_queries = num_track_queries_with_id = 0
    if tracking:
        num_track_queries = len(target['track_query_boxes'])
        num_track_queries_with_id = len(target['track_query_match_ids'])
        track_ids = target['track_ids'][target['track_query_match_ids']]

    keep = result['scores'].cpu() > result['scores_no_object'].cpu()

    cmap = get_hsv_color_map(len(keep))

    axs = display_images(axarr, img_groups, img_ids)

    process_and_visualize_boxes(
        axs[0],
        keep,
        result,
        target,
        tracking,
        track_ids,
        cmap
    )

    query_keep = keep
    if tracking:
        track_queries_mask = target['track_queries_mask'].to(keep.device)
        query_keep = keep[track_queries_mask == 0]

    legend_handles = create_legend_handles(
        target,
        keep,
        query_keep,
        num_track_queries,
        num_track_queries_with_id
    )
    axs[0].legend(handles=legend_handles)

    visualize_frame_targets(
        axs[2:],
        target,
        frame_prefixes=frame_prefixes
    )

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.axis('off')

    img = fig_to_numpy(figure).transpose(2, 0, 1)
    plt.close()

    visualizer.plot(img)

    tensor = features[0].tensors[0]

    visualizer.viz.images(
        NormalizeHelper.per_channel(tensor),
        nrow=1,
        opts={
            'title': '4to3_per_channel',
            'width': 1200, 
            'height': 1200,
        },
        win='4to3_per_channel',
    )

    visualizer.viz.images(
        NormalizeHelper.across_channels(tensor),
        nrow=1,
        opts={
            'title': '4to3_across',
            'width': 1200, 
            'height': 1200,
        },
        win='4to3_across',
    )



def build_visualizers(
    args: dict,
    train_loss_names: list
):
    visualizers = {}
    visualizers['train'] = {}
    visualizers['val'] = {}

    if args.eval_only or args.no_vis or not args.vis_server:
        return visualizers

    env_name = str(args.output_dir).split('/')[-1]

    vis_kwargs = {
        'env': env_name,
        'resume': args.resume and args.resume_vis,
        'port': args.vis_port,
        'server': args.vis_server
    }

    #
    # METRICS
    #

    legend = ['loss']
    legend.extend(train_loss_names)

    # for i in range(len(train_loss_names)):
    #     legend.append(f"{train_loss_names[i]}_unscaled")

    legend.extend([
        'class_error',
        # 'loss',
        # 'loss_bbox',
        # 'loss_ce',
        # 'loss_giou',
        # 'loss_mask',
        # 'loss_dice',
        # 'cardinality_error_unscaled',
        # 'loss_bbox_unscaled',
        # 'loss_ce_unscaled',
        # 'loss_giou_unscaled',
        # 'loss_mask_unscaled',
        # 'loss_dice_unscaled',
        'lr',
        'lr_backbone',
        'iter_time'
    ])

    opts = dict(
        title="TRAIN METRICS ITERS",
        xlabel='ITERS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend
    )

    # TRAIN
    visualizers['train']['iter_metrics'] = LineVis(opts, **vis_kwargs)

    opts = copy.deepcopy(opts)
    opts['title'] = "TRAIN METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    opts['legend'].remove('lr')
    opts['legend'].remove('lr_backbone')
    opts['legend'].remove('iter_time')
    visualizers['train']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = "VAL METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    visualizers['val']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    #
    # EVAL COCO
    #

    legend = [
        'BBOX AP IoU=0.50:0.95',
        'BBOX AP IoU=0.50',
        'BBOX AP IoU=0.75',
    ]

    if args.masks:
        legend.extend([
            'MASK AP IoU=0.50:0.95',
            'MASK AP IoU=0.50',
            'MASK AP IoU=0.75'])

    if args.tracking and args.tracking_eval:
        legend.extend(['MOTA', 'IDF1'])

    opts = dict(
        title='TRAIN EVAL EPOCHS',
        xlabel='EPOCHS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EVAL EPOCHS'
    visualizers['val']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    #
    # EXAMPLE RESULTS
    #

    opts = dict(
        title="TRAIN EXAMPLE RESULTS",
        width=2500,
        height=2500)

    # TRAIN
    visualizers['train']['example_results'] = ImgVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EXAMPLE RESULTS'
    visualizers['val']['example_results'] = ImgVis(opts, **vis_kwargs)

    return visualizers
