# -*- coding: cp1251 -*-
import sys
import os
import shutil
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPaintEngine, QPixmap, QIcon, QTransform
from PyQt6.QtCore import Qt
from PyQt6 import uic


class ImageAnalysisApp(QMainWindow):
    def __init__(self):
        # NEW CODE 21.01.24
        super().__init__()
        ui_file = "interface\ImgAnalysis.ui"
        self.ui_class, base_class = uic.loadUiType(ui_file)
        self.ui = self.ui_class()
        self.ui.setupUi(self)

        self.set_image()
        self.init_events()
        self.path = None
        self.path_result = None
        self.image_name = None
        self.model = YOLO('models/T2.pt')
        self.folder_path = "runs/detect/predict"
        self.zoom_delta = 1.2

        self.graphics_view = self.ui.graphicsView
        self.graphics_scene = QGraphicsScene()
        self.graphics_pixmap_item = None
        
    def set_image(self):
        self.showMaximized()
        self.setWindowIcon(QIcon("Icon\EYE.ico"))

    def update_image_analysis(self):
        if self.path:
            self.perform_image_analysis()

    def init_events(self):
        self.ui.horizontalSlider_Confidence.valueChanged.connect(
            lambda value: self.update_label_confidence(value, self.ui.label_Confidence)
        )
        self.ui.horizontalSlider_Overlap.valueChanged.connect(
            lambda value: self.update_label_confidence(value, self.ui.label_Overlap)
        )
        self.ui.Button_file.clicked.connect(self.choose_image_file)
        self.ui.action_open.triggered.connect(self.choose_image_file)
        self.ui.Button_update.clicked.connect(self.update_image_analysis)
        self.ui.action_save.triggered.connect(self.save_image_as)
        self.ui.widget_2.dragEnterEvent = self.on_drag_enter_event
        self.ui.widget_2.dropEvent = self.on_drop_event
        self.ui.widget_2.setAcceptDrops(True)

        self.ui.graphicsView.mousePressEvent = self.on_mouse_press_event
        self.ui.graphicsView.mouseMoveEvent = self.on_mouse_move_event

    def count_objects_in_labels_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            return line_count
        except (FileNotFoundError, IndexError):
            return 0

    def save_image_as(self):
        if self.path_result:
            default_file_name = f"ImAn_{self.image_name}"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", default_file_name,
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp)"
            )
            if file_path:
                if file_path == default_file_name:
                    default_file_name = os.path.basename(self.path_result)
                    file_path = os.path.join(os.path.dirname(file_path), default_file_name)
                shutil.copy2(self.path_result, file_path)

    def choose_image_file(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                caption="Select Image", filter="Image Files (*.png *.jpg *.jpeg *.tif)"
            )
        self.path = path
        self.perform_image_analysis()

    def perform_image_analysis(self):
        if self.path:
            shutil.rmtree("runs/detect/predict", ignore_errors=True)
            self.image_name = os.path.split(self.path)[-1]
            conf = self.ui.horizontalSlider_Confidence.value() / 100.0
            overlap = self.ui.horizontalSlider_Overlap.value() / 100.0
            show_conf = self.ui.checkBox_1.isChecked() 
            show_labels = self.ui.checkBox_3.isChecked()
            results = self.model(
                 self.path, show_conf=show_conf, show_labels=show_labels, 
                 conf=conf, iou=overlap, save=True, save_txt=True, max_det=600, line_width=1
             )
            self.path_result = f"{self.folder_path}/{self.image_name}"
            label_files = os.listdir(f'{self.folder_path}/labels')
            path_quantity = os.path.join(f'{self.folder_path}/labels', label_files[0]) if label_files else None
            quantity = self.count_objects_in_labels_file(path_quantity) if path_quantity else 0

            self.graphics_scene.clear()
            pixmap = QPixmap(self.path_result)
            self.graphics_pixmap_item = self.graphics_scene.addPixmap(pixmap)
            self.graphics_view.setScene(self.graphics_scene)
            self.graphics_view.fitInView(self.graphics_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

            self.ui.label_name.setText(self.image_name)
            self.ui.label_quantity.setText(f"{quantity} шт.")
            
            if self.path:
                self.ui.label_img2.setText("")
  

    def on_drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def on_drop_event(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            self.choose_image_file(path)

    def on_mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.pos()

    def on_mouse_move_event(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.pos() - self.start_pos
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() - delta.x()
            )
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() - delta.y()
            )

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.graphics_view.scale(self.zoom_delta, self.zoom_delta)
        else:
            self.graphics_view.scale(1 / self.zoom_delta, 1 / self.zoom_delta)
            
    def update_label_confidence(self, value, label):
        label.setText(f"{value}%")
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageAnalysisApp()
    window.show()
    sys.exit(app.exec())
