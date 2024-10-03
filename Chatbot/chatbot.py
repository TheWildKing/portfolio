import flet as ft  # pretty cool, huh? https://flet.dev/ | py -m pip install flet
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from RealtimeTTS import TextToAudioStream, CoquiEngine
from multiprocessing import Process, freeze_support
import speech_recognition as sr
import tracemalloc
import matplotlib.pyplot as plt
import torch
from diffusers import FluxPipeline
import numpy as np
import base64
from flet import Image
from io import BytesIO
from PIL import Image as imag

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


# main loop, aka every tick
async def main(page: ft.Page):
    page.title = "Chatbot"
    #page.vertical_alignment = ft.MainAxisAlignment.CENTER
    #page.scroll = ft.ScrollMode.ALWAYS
    tracemalloc.start()
    pb = ft.ProgressBar()

    # instructions for bot
    template = """
        If the user asks you to create an image, say "Generating image". 
        Otherwise,
        Answer the question below.

        Here is the conversation history: {context}

        Question: {question}
        
        Answer:
        """
    
    # set up bot
    model = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    engine = CoquiEngine()
    stream = TextToAudioStream(engine)

    # set up ui
    txt_label = ft.Text(value="Send a message. Say 'exit' to quit.", text_align=ft.TextAlign.CENTER)
    # txt_number = ft.TextField(value="", text_align=ft.TextAlign.CENTER, width=500)
    output_txt_label= ft.Text(value="", text_align=ft.TextAlign.CENTER, width=600, size=24)

    column_view = ft.Column(
        height = 550,
        width = 600,
        scroll = ft.ScrollMode.AUTO,
        spacing = 10,
    )

    page.add( 
        ft.Row(
            [
                column_view
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        
    )
 
    # this method is really bad... it causes errors, but it gets the job done. oh well. *shrugs*
    def window_event(e):
        if e.data == "close":       
            Process(target=handle_conversation).close() # i have no idea how this Process api works, you'd think it would forcefully stop the method but i think nothing happens
            # shut everything down!!!
            engine.shutdown()
            stream.stop()
            tracemalloc.stop()
            page.window.destroy() # *explosions* 
    
    page.window.prevent_close = True 
    page.window.on_event = window_event  

    async def stop_stream(e):
        print("skip request detected")
        #if stream.stream_running():
        stream.stop()

    # use PyAudio to record the user
    async def start_listening():
        path = "output.wav"
        await audio_rec.start_recording_async(path)  # is await needed?
            
    
    async def handle_conversation():
        global context
        context = ""
        page.add(pb)
        page.update()
        output_path = await audio_rec.stop_recording_async()
        
        r = sr.Recognizer()
        with sr.AudioFile(output_path) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            print("You: " + text)
            
        user_input = text
        
        # display text of user's message
        users_text = ft.Text(value="" + user_input, text_align=ft.TextAlign.CENTER, width = 400, color=ft.colors.WHITE)
        column_view.controls.append( 
            ft.Row([
                ft.Container(
                    content=users_text,
                    alignment=ft.alignment.center_left,
                    bgcolor=ft.colors.GREY_900,
                    padding=10,
                    border_radius=10,
                    margin=10,
                    width = users_text.width
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            )
        )

        page.update()

        # close everything if user says "exit"
        if user_input.lower() == "exit":
            engine.shutdown()
            stream.stop()
            tracemalloc.stop()
            page.window.destroy()

        # ~ai magic, commence!~
        result = chain.invoke({"context": context, "question": user_input})
        print("Bot:", result)
        context += f"\nUser: {user_input}\nAI: {result}"


        # display text of chatbot's message
        column_view.controls.append( 
        ft.Row(
            [
                ft.Container(
                    content=ft.Text(value="" + result, text_align=ft.TextAlign.CENTER, width=400, color=ft.colors.WHITE, no_wrap=False),
                    alignment=ft.alignment.center_right,
                    bgcolor=ft.colors.BLUE_900,
                    padding=10,
                    border_radius=10,
                    margin=10
                )
                
            ],
            alignment=ft.MainAxisAlignment.END,
            )
        )

        page.update()

        # play the audio stream of the chatbot
        stream.feed(result).play(log_synthesized_text=True)

        if "generating image" in result.lower():
            print("image generation request received")
            prompt = user_input
            print ("Prompt: " + prompt + " | Bot: " + result)
            """if "generating image" in result.lower():
                global most_recent_user_prompt
                most_recent_user_prompt = prompt
            elif "recreatimg image" in result.lower():
                prompt = user_input + "most_recent_image_prompt + (previous image prompt: {most_recent_user_prompt})"
                most_recent_user_prompt = prompt"""  # image regeneration corrupts program, must find fix later
            image = pipe(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            image.save("flux.png")


            # credit to VicourtBitt on Stack Overflow. https://stackoverflow.com/questions/77592627/how-to-update-an-image-frame-displayed-in-flet
            
            image_path = r"flux.png" # First Reachable Path
            pil_photo = imag.open(image_path) # Pillow Opens the Image
            arr = np.asarray(pil_photo) # Numpy transforms it into an array

            pil_img = imag.fromarray(arr) # Then you convert it in an image again
            buff = BytesIO() # Buffer
            pil_img.save(buff, format="PNG") # Save it

            image_string = base64.b64encode(buff.getvalue()).decode('utf-8')
            image1 = Image(src_base64=image_string,width=300,height=300,fit=ft.ImageFit.CONTAIN, border_radius=10)

            newstring = base64.b64encode(buff.getvalue()).decode("utf-8")
            image1.src_base64 = newstring

            """img = ft.Image(
            src=image1,
            width=300,
            height=300,
            fit=ft.ImageFit.CONTAIN,
            )"""

            image_label = ft.Row([image1], alignment=ft.MainAxisAlignment.END,)

            column_view.controls.append(image_label)

            # image1.update() # "Voí'la"
            page.update()

        page.remove(pb)
    
        page.update()

    async def handle_state_change(e):
        print(f"State Changed: {e.data}")

    audio_rec = ft.AudioRecorder(
        audio_encoder=ft.AudioEncoder.WAV,
        on_state_changed=handle_state_change,
    )
    page.overlay.append(audio_rec)
    await page.update_async()

    async def toggle_icon_button(e):
        print(e.control.selected)
        if e.control.selected == False:
            e.control.selected = not e.control.selected
            e.control.update()
            await start_listening()
        else:
            e.control.selected = not e.control.selected
            e.control.update()
            page.run_task(handle_conversation)

        
        e.control.update()

    # display ui, sequentially add rows
    page.add( 
        ft.Row(
            [
                txt_label,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
 
    )
    page.add( 
        ft.Row(
            [
                ft.IconButton(
                    icon=ft.icons.MIC,
                    selected_icon=ft.icons.STOP,
                    on_click=toggle_icon_button,
                    selected=False,
                    style=ft.ButtonStyle(color={"selected": ft.colors.RED, "": ft.colors.GREEN}),
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    
    )

    


if __name__ == '__main__':
    freeze_support() # how on earth does this solve the freezing problem?! why is it so easy? its almost too good to be true... surely is CPU heavy?
    ft.app(target=main)