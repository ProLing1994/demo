import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
from easydict import EasyDict as edict


def build_hist(x, name, start, end, stride, color='#FF0000', opacity=1.0):

    trace = go.Histogram(x=x, name=name,
                         xbins=dict(start=start, end=end, size=stride),
                         marker=dict(color=color), opacity=opacity)
    return trace


def hist_comp(hists, out_file, title='histograms', xname='Value', yname='Count',
              bargap=0.2, bargroupgap=0.1, auto_open=False):
    """
    generate a figure with multiple histograms
    :param hists: a list of histogram objects
    :param out_file: output file path
    :param title: the figure title
    :param xname: the name of x axis
    :param yname: the name of y axis
    :param bargap: the gap between bars
    :param bargroupgap: the gap between bar groups
    :param auto_open: automatically open figure when save
    :return: None
    """
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xname),
        yaxis=dict(title=yname),
        bargap=bargap,
        bargroupgap=bargroupgap
    )
    fig = go.Figure(data=hists, layout=layout)
    py.plot(fig, filename=out_file, auto_open=auto_open)


def plot_loss(log_file, out_file, name='dloss', display='Dice loss'):
    """
    For the consideration of compatibility.
    """
    loss_names = [name]
    legends = [display]
    colors = ['rgb(255, 0, 0)']
    plot_losses(log_file, out_file, display, loss_names, legends, colors)


def plot_loss2d(log_file, out_file, name=['train_loss', 'test_loss'], display='Dice loss'):
    """
    For the consideration of compatibility.
    """
    loss_names = name
    legends = name
    colors = ['rgb(255, 0, 0)', 'rgb(0, 0, 255)']
    plot_losses(log_file, out_file, display, loss_names, legends, colors)


def plot_loss4d(log_file, out_file, name=['model_0_train_loss', 'model_1_train_loss', 'model_0_eval_loss', 'model_1_eval_loss'], display='Dice loss'):
    """
    For the consideration of compatibility.
    """
    loss_names = name
    legends = name
    colors = ['rgb(255, 0, 0)', 'rgb(255, 127, 0)', 'rgb(0, 0, 255)', 'rgb(0, 127, 255)']
    plot_losses(log_file, out_file, display, loss_names, legends, colors)


def plot_losses(log_file, out_file, display, loss_names, legends, colors=None, batch_word ='batch'):
    """
    Plot loss curve from the data in log_file
    :param log_file:    Log file
    :param out_file:    Output plot html file
    :param display:     Message shown on the top of the plot
    :param loss_names:  A list of loss data to be plotted.
    :param legends:     A list of legend with respect to loss_names.
    :param colors:      A list of colors with respect to loss_names, in the form of ['rgb(a,b,c)','rgb(d,e,f)'...]
    :param batch_word:  Keyword to help locate the batch index in log file, which will be the x-axis of the plot.
    :return: None.
    """
    assert len(loss_names) == len(legends), "the length of loss_names and displays do not match!"
    batch_idxs_dict = edict.fromkeys(loss_names, [])
    loss_values_dict = edict.fromkeys(loss_names, [])
    batch_word = batch_word + ':'
    num_loss = len(loss_names)

    # if the colors are not specified, use random colors.
    if colors is None:
        colors = []
        color_grid = np.random.randint(0, 255, [num_loss, 3])
        for i in range(num_loss):
            color = 'rgb({0},{1},{2})'.format(color_grid[i, 0],
                                            color_grid[i, 1],
                                            color_grid[i, 2])
            colors.append(color)

    with open(log_file, 'r') as f:
        for line in f:
            if not batch_word in line:
                continue
            for loss_name in loss_names:
                try:
                    loss_word = loss_name + ':'
                    if loss_word in line:
                        batch_idx, loss_value = extract_point_from_message(
                            line, batch_word, loss_word)
                        batch_idxs_dict[loss_name].append(batch_idx)
                        loss_values_dict[loss_name].append(loss_value)
                except:
                    continue

    # plot curve and save it to html
    traces = []
    for trace_idx, loss_name in enumerate(loss_names):

        trace = generate_trace(batch_idxs_dict[loss_name], loss_values_dict[loss_name],
                           trace_name=legends[trace_idx], color=colors[trace_idx])
        traces.append(trace)

    layout = dict(title=display + ' during training',
                  xaxis=dict(title='Batch iteration'),
                  yaxis=dict(title=display))

    fig = dict(data=traces, layout=layout)
    py.plot(fig, filename=out_file, auto_open=False)


def extract_point_from_message(line, batch_word, loss_word):
    """
    Extract information that make up a point of a trace from one line of the log file.
    :param line: One line from the log file.
    :param batch_word: A string, keyword to locate the batch index.
    :param loss_word: A string, keyword to locate the loss value.
    :return: One point of a trace
    """
    # locate the batch index.
    start = line.find(batch_word) + len(batch_word)
    end = line.find(',', start)
    batch_idx = float(line[start:end])
    # locate the loss value.
    start = line.find(loss_word) + len(loss_word)
    end = line.find(',', start)
    loss_value = float(line[start:end])

    return batch_idx, loss_value


def generate_trace(batch_idxs, loss_values, trace_name, color='rgb(0, 0, 0)', width=1):
    """
    Generates a trace and put it in a traces list.
    :param batch_idxs: A list containing the coordinates of the x-axis of a trace.
    :param loss_values: A list containing the coordinates of the y-axis of a trace.
    :param trace_name: A string, the name of a trace, will be showed in the legend.
    :param color: A string, defining the color of a trace line.
    :param width: A number, the line width of a trace line.
    :return: A trace consists of X-Y data that can be plotted by py.plot.
    """
    linetype = dict(color=color, width=width)
    trace = go.Scatter(
        x=np.array(batch_idxs, dtype=np.float),
        y=np.array(loss_values, dtype=np.float),
        name=trace_name,
        line=linetype
        )
    return trace
