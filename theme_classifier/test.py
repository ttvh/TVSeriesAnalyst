import gradio as gr
import pandas as pd

def plot_chart():
    data = {"Theme": ["Friendship", "Hope", "Sacrifice", "Battle", "Self Development", "Betrayal", "Love"],
            "Score": [80, 120, 150, 130, 100, 90, 50]}
    
    df = pd.DataFrame(data)

    return gr.BarPlot(
        df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme", "Score"],
        vertical=False,  # Biểu đồ ngang
        width=500,
        height=300
    )

with gr.Blocks() as demo:
    gr.HTML("<h2>Test Gradio Bar Plot</h2>")
    plot = gr.BarPlot()
    btn = gr.Button("Generate Chart")
    btn.click(plot_chart, outputs=[plot])

demo.launch(share=True)
