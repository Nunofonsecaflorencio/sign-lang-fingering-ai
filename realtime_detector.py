import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure
import cv2
import time

from preprocessing import *

def cv2_to_bytes(img):
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

    
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def get_top_k_indexes(prediction, k):
    # Get the indices of the top k elements
    k = min(len(prediction), k)
    return prediction.argsort()[-k:][::-1]

def update_plot(ax, x, y):
    ax.cla()                    # clear the subplot
    ax.grid(False)              # draw the grid
    
    bars = ax.bar(x, y, width=.6)
    
    # Add text representing percentage and x-value on top of each bar
    for xi, yi, bar in zip(x, y, bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}', ha='center', va='bottom', weight='bold', color='purple')
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{xi}', ha='center', va='top', weight='bold', color='yellow')
    
    ax.axis('off')
    ax.set_title("Nível de Confiaça do Modelo")


def main():

    # define the form layout
    layout = [[sg.Stretch(), sg.Text('Sign Language Fingering Detection', justification='center', font='Helvetica 20'), sg.Stretch()],
              [sg.Image(filename='', key='-IMAGE-', expand_x=True), sg.Canvas(key='-CANVAS-')],
              [sg.Multiline(size=(None, 3), font='Helvetica 28 bold',  key='-PREDICTION-')],
              [sg.Button('Limpar', size=(16, 2)), sg.Stretch(), sg.Button('Sair', button_color='red', size=(16, 2))]]

    # create the form and show it without the plot
    window = sg.Window('Sign Language Fingering AI', 
                layout, finalize=True, size=(1000, 720), font='Helvetica 14')
    cap = cv2.VideoCapture(0)
    
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas

    # draw the initial plot in the window
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Action")
    ax.set_ylabel("Confidence")
    ax.set_title("Nível de Confiaça do Modelo")
    ax.axis('off')
    # ax.grid()
    fig_agg = draw_figure(canvas, fig)
    
    model, labels = load_signmodel() 
    output_text = ''
    timestamp = 0
    TIME_PER_PREDICTION = 1
    start_prediction_time = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:

            event, values = window.read(timeout=50)
            if event in ('Sair', None):
                exit()
            if event == 'Limpar':
                output_text = ''
                window['-PREDICTION-'].update(output_text)
            
            ret, frame = cap.read()
            if not ret:
                sg.popup_error('Error reading from camera!')
                break
                    
            data, annotated_frame = process_frame(frame, landmarker, timestamp)    
            
            current_predition_time = time.time()
            dt = current_predition_time - start_prediction_time
            if dt > TIME_PER_PREDICTION:
                start_prediction_time = current_predition_time
                
                if data.max():
                    prediction = model.predict(np.array([data])).flatten()
                    predicted_action = labels[prediction.argmax()].upper()
                    
                    top_actions_idx = get_top_k_indexes(prediction, 3)
                    top_actions = [labels[i].upper() for i in top_actions_idx]
                    confidence_values = [prediction[i] for i in top_actions_idx]
                    
                    output_text += predicted_action
                    update_plot(ax, top_actions, confidence_values)
                   
                
                elif output_text and output_text[-1] != ' ':
                    output_text += ' '

                window['-PREDICTION-'].update(output_text)
                
                
            # Display OpenCV camera feed
            window['-IMAGE-'].update(data=cv2_to_bytes(annotated_frame))    
            
            
            fig_agg.draw()
            timestamp += 1

    window.close()

if __name__ == '__main__':
    main()