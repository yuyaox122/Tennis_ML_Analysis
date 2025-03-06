from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.video import Video
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.uix.label import Label
import os

class VideoPlayerApp(App):
    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical')

        # Video player widget
        self.video = Video(source="long_clip.mp4", state='stop', options={'eos': 'loop'})
        layout.add_widget(self.video)

        # Controls layout (horizontal)
        controls = BoxLayout(size_hint_y=0.2)

        # Play button
        play_button = Button(text="Play", on_press=self.play_video)
        controls.add_widget(play_button)

        # Pause button
        pause_button = Button(text="Pause", on_press=self.pause_video)
        controls.add_widget(pause_button)

        # Stop button
        stop_button = Button(text="Stop", on_press=self.stop_video)
        controls.add_widget(stop_button)

        # File select button
        file_button = Button(text="Select File", on_press=self.open_filechooser)
        controls.add_widget(file_button)

        # Seek slider
        self.slider = Slider(min=0, max=100, value=0)
        self.slider.bind(on_touch_up=self.seek_video)
        controls.add_widget(self.slider)

        layout.add_widget(controls)

        # Update slider position periodically
        Clock.schedule_interval(self.update_slider, 0.5)

        return layout

    def play_video(self, instance):
        # Play the video
        if self.video.state != 'play':
            self.video.state = 'play'

    def pause_video(self, instance):
        # Pause the video
        if self.video.state == 'play':
            self.video.state = 'pause'

    def stop_video(self, instance):
        # Stop the video and reset to the beginning
        self.video.state = 'stop'
        self.video.seek(0)
        self.slider.value = 0

    def update_slider(self, dt):
        # Update the slider based on video progress
        if self.video.duration and self.video.state == 'play':
            self.slider.max = self.video.duration
            self.slider.value = self.video.position

    def seek_video(self, instance, touch):
        # Seek the video when the slider is moved
        if instance.collide_point(*touch.pos):
            self.video.seek(instance.value)

    def open_filechooser(self, instance):
        # Open file chooser dialog
        content = FileChooserIconView()
        content.bind(on_selection=lambda widget, selection: self.selected(selection))

        self.popup = Popup(title="Select a Video File", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def selected(self, selection):
        # Callback for file selection
        if selection:
            video_file = selection[0]
            self.video.source = video_file
            self.video.state = 'stop'
            self.video.seek(0)
            self.slider.value = 0
            self.popup.dismiss()
            print(f"Selected file: {video_file}")

if __name__ == "__main__":
    VideoPlayerApp().run()
