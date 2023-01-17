"""
This python script is for an interactive GUI for visualizating the tree structure of the topics.
"""

import tkinter
import tkinter.ttk as ttk
import webbrowser
import pandas as pd
import os
import time
import math

# this is data.py
import data

# window configurations
window = tkinter.Tk()
window.title("Learning Equality competition topics tree structure visualizer")
# window.resizable(False, False)
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = 1280
window_height = 720
window.geometry("{}x{}+{}+{}".format(window_width, window_height, int((screen_width - window_width)/2.0), int((screen_height - window_height)/2.0)))
window.minsize(1000, 500)

# add some buttons and stuff to the window

# bottom side (buttons)
bottom_frame = ttk.Frame(window)
bottom_frame.pack(side = tkinter.BOTTOM, fill = tkinter.X)

# set up scrolling for bottom side buttons
bottom_scroll = ttk.Scrollbar(bottom_frame, orient='horizontal')
bottom_scroll.pack(side = tkinter.BOTTOM, fill = tkinter.X)
bottom_scroll_view = tkinter.Canvas(bottom_frame, xscrollcommand = bottom_scroll.set)
bottom_scroll_view.pack(side = tkinter.TOP, fill = tkinter.BOTH)
bottom_button_container = ttk.Frame(bottom_scroll_view)
# trigger resizing whenever the bottom_button_container resizes
def reconfigure_scroll_view(x):
    bounding_box = bottom_scroll_view.bbox("all")
    bottom_scroll_view.configure(height = bounding_box[3] - bounding_box[1])
    # set scroll region to all of the canvas size
    bottom_scroll_view.configure(scrollregion = bounding_box)
bottom_button_container.bind("<Configure>", reconfigure_scroll_view)
# add bottom_button_container to the scroll view
bottom_scroll_view.create_window(0, 0, window = bottom_button_container, anchor = "nw")
# dummy function, similar to c++ declaration and definition.
def select_tree(tree_id):
    pass

k = 0
btn_style = ttk.Style()
btn_style.configure("TButton", anchor = "center")
btn_style.configure("TButton", justify = "center")
tree_buttons = []
for topic_channel in data.topic_trees_id_list:
    btnk = ttk.Button(bottom_button_container, text = data.topic_trees[topic_channel].title + "\nChannel: " + data.topic_trees[topic_channel].channel, style = "TButton")
    btnk.grid(row = 0, column = k, padx = 4)
    def fix_lambda(topic_channel):
        return lambda: select_tree(topic_channel)
    btnk.configure(command = fix_lambda(topic_channel))
    k += 1
    tree_buttons.append(btnk)
bottom_scroll.config(command = bottom_scroll_view.xview)

# end of bottom side

# top side
top_side = ttk.Frame(window)
top_side.pack(fill = tkinter.BOTH, expand = True)
search_btn = ttk.Button(top_side, text = "Search for trees / nodes by name / uid")
search_btn.pack(fill = tkinter.X, side = tkinter.BOTTOM, padx = 4)
top_frame = ttk.Frame(top_side)
top_frame.pack(fill = tkinter.BOTH, expand = True)
# tree view in right
tree_frame = ttk.Frame(top_frame)
tree_frame.pack(side = tkinter.RIGHT, fill = tkinter.Y, padx = 10, pady = 10)
tree_frame.configure(width = 350)
tree_frame.pack_propagate(False)
tree_structure_view = ttk.Treeview(tree_frame)
horizontal_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
horizontal_scroll.pack(side = tkinter.BOTTOM, fill = tkinter.X)
vertical_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
vertical_scroll.pack(side = tkinter.RIGHT, fill = tkinter.Y)
tree_structure_view.configure(xscrollcommand = horizontal_scroll.set)
tree_structure_view.configure(yscrollcommand = vertical_scroll.set)
tree_structure_view.pack(fill = tkinter.BOTH, expand = True)
tree_structure_view.column('#0', width = 1200)
horizontal_scroll.config(command = tree_structure_view.xview)
vertical_scroll.config(command = tree_structure_view.yview)

# define events for tree viewer
def on_node_toggle(evt):
    node = get_selected_node_on_tree()
    mrenderer.expand_tree_external(node)
    mrenderer.render_scene()

def on_node_click(evt):
    def gen_lambda(val):
        return (lambda x: x.uid == val)
    nodeid = tree_structure_view.identify_row(evt.y)
    node = mrenderer.active_tree.iterate_find(gen_lambda(nodeid))
    if node is not None:
        mrenderer.navigate_to(node.get_x(), node.get_y())
        tree_structure_view.focus(node.uid)
        mrenderer.render_scene()

tree_structure_view.bind("<<TreeviewOpen>>", on_node_toggle)
tree_structure_view.bind("<<TreeviewClose>>", on_node_toggle)
tree_structure_view.bind("<Button-1>", on_node_click)

def get_selected_node_on_tree():
    def gen_lambda(val):
        return (lambda x: x.uid == val)
    fcs = tree_structure_view.focus()
    if fcs is None:
        return None
    node = mrenderer.active_tree.iterate_find(gen_lambda(fcs))
    return node
# renderer in the left
node_track_pos = 0
class RendererNode:
    def __init__(self, title, description, channel, category, level, language, has_content, uid, display_id):
        self.parent = None
        self.children = []
        self.title = title
        self.description = description
        self.channel = channel
        self.category = category
        self.level = level
        self.language = language
        self.has_content = has_content
        self.uid = uid

        self.expanded = False
        self.occupy_space_last_compute = 0
        self.rel_x_pos_last_compute = 0
        self.rel_subtree_x_left_last_compute = 0
        self.order_last_compute = 0
        self.display_id = display_id

    def __del__(self):
        for child in self.children:
            del child
        del self.children

    def occupy_space(self):
        if self.expanded:
            count = 0
            for child in self.children:
                count += child.occupy_space()
            if count == 0:
                self.occupy_space_last_compute = 1
                return 1
            self.occupy_space_last_compute = count
            return count
        else:
            self.occupy_space_last_compute = 1
            return 1

    def generate_relative_positions(self, start_val = 0, shift_factor = 0):
        global node_track_pos
        if self.level == 0:
            node_track_pos = 0
        self.order_last_compute = node_track_pos
        node_track_pos += 1
        if self.level == 0:
            self.rel_subtree_x_left_last_compute = 0
            shift_factor = -(self.occupy_space_last_compute - 1) / 2.0
        else:
            self.rel_subtree_x_left_last_compute = start_val
        self.rel_x_pos_last_compute = self.rel_subtree_x_left_last_compute + (self.occupy_space_last_compute - 1) / 2.0 + shift_factor
        if self.expanded:
            for child in self.children:
                child.generate_relative_positions(start_val, shift_factor = shift_factor)
                start_val += child.occupy_space_last_compute

    def get_y(self):
        return self.level * 300

    def get_x(self):
        return self.rel_x_pos_last_compute * 300

    def iterate_find_on_expanded(self, condition):
        if condition(self):
            return self
        if self.expanded:
            for child in self.children:
                res = child.iterate_find_on_expanded(condition)
                if res is not None:
                    return res
        return None

    def iterate_find(self, condition):
        if condition(self):
            return self
        for child in self.children:
            res = child.iterate_find(condition)
            if res is not None:
                return res
        return None

node_copier_idx = -1
def node_copier(node):
    global node_copier_idx
    node_copier_idx += 1
    return RendererNode(node.title, node.description, node.channel, node.category, node.level, node.language, node.has_content, node.uid, node_copier_idx)

class LeftRenderer:
    def __init__(self):
        self.renderer = tkinter.Canvas(top_frame)
        self.renderer.pack(fill = tkinter.BOTH, expand = True, padx = 10, pady = 10)
        self.renderer.config(background = "white")
        self.scale = 1.0
        self.renderer.bind("<Button-1>", self.mouse_left_click)
        self.renderer.bind("<Double-3>", self.mouse_right_click)
        self.renderer.bind("<B1-Motion>", self.mouse_drag)
        self.renderer.bind("<MouseWheel>", self.mouse_scroll)
        self.active_tree = None

        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.last_mouse_click_time = -10000
        self.x_top = -self.renderer.winfo_width() / (2.0 * self.scale)
        self.y_top = -self.renderer.winfo_height() / (2.0 * self.scale)

    def refresh_tree(self, tree_id):
        global node_copier_idx
        node_copier_idx = -1
        self.active_tree = data.topic_trees[tree_id].deep_copy_equivalent(node_copier)
        self.active_tree.occupy_space()
        self.active_tree.generate_relative_positions()
        self.x_top = -self.renderer.winfo_width() / (2.0 * self.scale)
        self.y_top = -self.renderer.winfo_height() / (2.0 * self.scale)

    def render_scene(self):
        self.renderer.delete("all")
        self.renderer.create_text(100, 30, text = "Left double click to expand nodes")
        self.renderer.create_text(100, 60, text="Drag the screen with mouse to move")
        self.renderer.create_text(100, 90, text="Scroll with mouse to magnify")
        self.renderer.create_text(self.renderer.winfo_width() - 150, 30, text="Right double click on bold circle nodes to view contents")
        self.renderer.create_text(self.renderer.winfo_width() - 150, 60, text="Left single click to select nodes")
        self.renderer.create_text(self.renderer.winfo_width() - 150, 90, text="Use search button below to search for nodes / tree names")
        if self.active_tree is not None:
            self.renderer.create_text(100, 120, text=self.active_tree.channel)
            self.render_node(self.active_tree, get_selected_node_on_tree())
        else:
            self.renderer.create_text(int(self.renderer.winfo_width() / 2.0), 60,
                                      text="Select a tree by\nclicking on one\nof the buttons\nat the bottom", font=('Arial',20,'bold italic'))

    def render_node(self, node, sel_node):
        x = node.get_x()
        y = node.get_y()
        shade = False
        if sel_node is not None:
            if sel_node.uid == node.uid:
                shade = True

        if node.has_content:
            if shade:
                self.draw_circle(x, y, 100, 100, width = 3, fill = "#99ffff")
            else:
                self.draw_circle(x, y, 100, 100, width = 3)
        else:
            if shade:
                self.draw_circle(x, y, 100, 100, width = 1, fill = "#99ffff")
            else:
                self.draw_circle(x, y, 100, 100, width = 1)
        if self.scale > 0.7:
            self.draw_text(x, y, node.title)
        else:
            self.draw_text(x, y, node.display_id)
        if node.expanded:
            if len(node.children) > 0:
                self.draw_line(x, y+50, x, y+150)
            for child in node.children:
                self.render_node(child, sel_node)
                child_x = child.get_x()
                self.draw_line(child_x, y + 150, child_x, y + 250)
            if len(node.children) > 1:
                self.draw_line(node.children[0].get_x(), y + 150, node.children[len(node.children)-1].get_x(), y + 150)

    def to_screen_pos(self, x, y):
        return (x-self.x_top)*self.scale, (y-self.y_top)*self.scale

    def to_real_pos(self, x, y):
        return (x + 0.0)/self.scale + self.x_top, (y + 0.0)/self.scale + self.y_top

    def navigate_to(self, x, y):
        rwidth = self.renderer.winfo_width()
        rheight = self.renderer.winfo_height()
        self.x_top = x - rwidth / (2 * self.scale)
        self.y_top = y - rheight / (2 * self.scale)

    def draw_circle(self, x, y, w, h, width = 1, fill = None):
        self.renderer.create_oval(int((x-w/2.0-self.x_top)*self.scale), int((y-h/2.0-self.y_top)*self.scale), int((x+w/2.0-self.x_top)*self.scale), int((y+h/2.0-self.y_top)*self.scale), width = width, fill = fill)

    def draw_text(self, x, y, text):
        self.renderer.create_text(int((x-self.x_top)*self.scale), int((y-self.y_top)*self.scale), text = text)

    def draw_line(self, x1, y1, x2, y2, width = 1):
        self.renderer.create_line(int((x1-self.x_top)*self.scale), int((y1-self.y_top)*self.scale), int((x2-self.x_top)*self.scale), int((y2-self.y_top)*self.scale), width = width)

    def mouse_left_click(self, evt):
        self.last_mouse_x = evt.x
        self.last_mouse_y = evt.y
        cur_time = (time.perf_counter_ns() // 1000000)
        if cur_time - self.last_mouse_click_time < 300:
            rx, ry = self.to_real_pos(evt.x, evt.y)
            if self.active_tree is not None:
                node = self.active_tree.iterate_find_on_expanded(lambda x: math.hypot(x.get_x() - rx, x.get_y() - ry) < 50)
                if node is not None:
                    self.expand_tree(node)
                    self.render_scene()
            cur_time = -10000
        self.last_mouse_click_time = cur_time

    def mouse_right_click(self, evt):
        rx, ry = self.to_real_pos(evt.x, evt.y)
        if self.active_tree is not None:
            node = self.active_tree.iterate_find_on_expanded(lambda x: math.hypot(x.get_x() - rx, x.get_y() - ry) < 50)
            if node is not None:
                if node.has_content:
                    contents = data.correlations.loc[node.uid]["content_ids"]
                    sframe = data.contents.loc[contents.split(" ")]

                    display_data_frame(sframe)

    def mouse_drag(self, evt):
        dx = evt.x - self.last_mouse_x
        dy = evt.y - self.last_mouse_y
        self.last_mouse_x = evt.x
        self.last_mouse_y = evt.y
        self.x_top -= dx / self.scale
        self.y_top -= dy / self.scale
        self.render_scene()

    def mouse_scroll(self, evt):
        bx = (self.x_top + evt.x / self.scale)
        by = (self.y_top + evt.y / self.scale)
        if evt.delta > 0:
            self.scale *= 1.1
        elif evt.delta < 0:
            self.scale /= 1.1
        self.x_top = bx - evt.x / self.scale
        self.y_top = by - evt.y / self.scale
        self.render_scene()

    def expand_tree_external(self, node):
        node.expanded = not node.expanded
        self.active_tree.occupy_space()
        self.active_tree.generate_relative_positions()

    def expand_tree(self, node):
        self.expand_tree_external(node)
        tree_structure_view.item(node.uid, open=node.expanded)
        tree_structure_view.selection_set(node.uid)
        tree_structure_view.focus(node.uid)
        global node_track_pos
        tree_structure_view.yview_moveto((node.order_last_compute+0.0) / node_track_pos)

    def expand_up_till(self, node):
        # expand parent nodes
        cur_node = node.parent
        while cur_node is not None:
            cur_node.expanded = True
            tree_structure_view.item(cur_node.uid, open=True)
            cur_node = cur_node.parent
        # select self node
        tree_structure_view.selection_set(node.uid)
        tree_structure_view.focus(node.uid)
        self.active_tree.occupy_space()
        self.active_tree.generate_relative_positions()
        global node_track_pos
        tree_structure_view.yview_moveto((node.order_last_compute + 0.0) / node_track_pos)
        # redraw scene
        self.navigate_to(node.get_x(), node.get_y())
        self.render_scene()

mrenderer = LeftRenderer()
mrenderer.render_scene()

# select tree
def select_tree(tree_id):
    if not tree_id in data.topic_trees:
        raise Exception("No such tree.")
    tree_structure_view.delete(*tree_structure_view.get_children())
    root = data.topic_trees[tree_id]

    idx = 0
    def recursion(current_node):
        nonlocal idx
        if current_node.parent is None:
            tree_structure_view.insert(parent="", index="end", id=current_node.uid, text=current_node.title + "  [" + str(idx) + "]")
        else:
            tree_structure_view.insert(parent=current_node.parent.uid, index="end", id=current_node.uid,
                                       text=current_node.title + "  [" + str(idx) + "]")
        idx += 1
        for child_node in current_node.children:
            recursion(child_node)

    recursion(root)
    tree_structure_view.xview_moveto(0)
    tree_structure_view.yview_moveto(0)
    mrenderer.refresh_tree(tree_id)
    mrenderer.render_scene()

# end of top side
# end of adding buttons and stuff

# function for searching for tree
def on_search_click():
    small_window = tkinter.Toplevel(window)
    small_window.title("Search for tree / node")
    winx, winy = window.winfo_rootx(), window.winfo_rooty()
    winwidth, winheight = window.winfo_width(), window.winfo_height()
    sw_width, sw_height = 800, 400
    small_window.geometry("{}x{}+{}+{}".format(sw_width, sw_height, winx + int((winwidth - sw_width)/2.0), winy + int((winheight - sw_height)/2.0)))
    tree_or_node = tkinter.StringVar(small_window)
    tree_or_node.set("Tree")
    option_menu = None
    if mrenderer.active_tree is not None:
        option_menu = ttk.OptionMenu(small_window, tree_or_node, "Tree", "Tree", "Node")
    else:
        option_menu = ttk.OptionMenu(small_window, tree_or_node, "Tree", "Tree")
    option_menu.pack(side = tkinter.TOP, fill = tkinter.X)
    search_text = ttk.Entry(small_window)
    confirm_search_btn = ttk.Button(small_window, text = "Search")
    confirm_search_btn.pack(side = tkinter.BOTTOM, fill = tkinter.X)
    search_text.pack(fill = tkinter.BOTH, expand = True)

    def on_search_window_click():
        nonlocal small_window, tree_or_node, search_text
        global tree_buttons
        # we search for title and then search or ID
        text = search_text.get()
        if text is not None and len(text) > 0:
            if tree_or_node.get() == "Tree":
                for k in range(len(tree_buttons)):
                    if text in tree_buttons[k]["text"]:
                        bounding_box = bottom_scroll_view.bbox("all")
                        bsv_width = bounding_box[2] - bounding_box[0]
                        bottom_scroll_view.xview_moveto((tree_buttons[k].winfo_x() + 0.0) / bsv_width)
            else:
                if mrenderer.active_tree is not None:
                    def mcondition(node):
                        nonlocal text
                        return text in node.title or text in node.uid
                    found_node = mrenderer.active_tree.iterate_find(mcondition)
                    if found_node is not None:
                        mrenderer.expand_up_till(found_node)

        small_window.destroy()
    confirm_search_btn.configure(command = on_search_window_click)
    small_window.grab_set()

search_btn.configure(command = on_search_click)

def display_data_frame(dframe):
    small_window = tkinter.Toplevel(window)
    # initiate contents
    tree_structure_view = ttk.Treeview(small_window, columns = tuple(dframe.columns))
    horizontal_scroll = ttk.Scrollbar(small_window, orient="horizontal")
    horizontal_scroll.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    vertical_scroll = ttk.Scrollbar(small_window, orient="vertical")
    vertical_scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    tree_structure_view.configure(xscrollcommand=horizontal_scroll.set)
    tree_structure_view.configure(yscrollcommand=vertical_scroll.set)
    tree_structure_view.pack(fill=tkinter.BOTH, expand=True)

    horizontal_scroll.config(command=tree_structure_view.xview)
    vertical_scroll.config(command=tree_structure_view.yview)

    for col in dframe.columns:
        tree_structure_view.heading(col, text = col)
    for idx in dframe.index:
        tree_structure_view.insert(parent = "", index = "end", id = str(idx), values = tuple(dframe.loc[idx]), text = str(idx))
    small_window.grab_set()

def open_temp_html(dframe):
    htmlstr = dframe.to_html()
    tempfile = os.getcwd() + "\\test.html"
    with open(tempfile, "w", encoding='utf8') as f:
        f.write(("<html>"+str(htmlstr)+"</html>"))

    webbrowser.open("file:///"+tempfile)
# show window
last_window_width = 0
last_window_height = 0
def on_resize(evt):
    global last_window_width, last_window_height
    if (last_window_width != evt.width) or (window_height != evt.height):
        last_window_width = evt.width
        last_window_height = evt.height
        mrenderer.render_scene()
window.bind("<Configure>", on_resize)
window.mainloop()