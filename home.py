import streamlit as st
from PIL import Image
import webbrowser
def app():
	st.title('Diabetes')
	st.text('Diabetes is a chronic (long-lasting) health condition that affects how your body turns food🍫 into energy🔋.\n\nMost of the food you eat is broken down into sugar 🍯(also called glucose) and released into your bloodstream🩸. When your\nblood sugar🎂 goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into\nyour body’s cells for use as energy.\n\nIf you have diabetes, your body either doesn’t make enough insulin or can’t use the insulin it makes as well as it should.\nWhen there isn’t enough insulin or cells stop responding\nto insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as\nheart disease 🫀, vision loss 👓, and kidney disease.\n\nThere isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in\nreducing the impact of diabetes.')
	image = Image.open('diabetes_img.jpg')
	st.image(image, caption='Diabetes')
	st.header('Ayurvedic Remedies for Diabetes')
	st.subheader('Triphala, Fenugreek and Shilajit')
	img2 = Image.open('trifenujit.png')
	st.image(img2)
	st.text('''Based on the available evidence, fenugreek has benefits for lowering blood sugar levels. Fenugreek may also reduce\ncholesterol levels, lower inflammation, and help with appetite control.\nTriphala, a combination of three fruits- haritaki, bibhitaki and amla,\ncontain many antioxidants. Antioxidants can help fight free\nradicals in the body, reducing inflammation along with your risk of\nchronic diseases like diabetes, heart disease, and others. Shilajit is a sticky\nsubstance found primarily in the rocks of the Himalayas.\nIt develops over centuries from the slow decomposition of plants. It can function\nas an antioxidant to improve your body's immunity\nand memory, an anti-inflammatory, an energy booster, and a diuretic\nto remove excess fluid from your body. Their mixture is a miracle for diabetes patients.''')
	if st.button('Order now'):
		webbrowser.open_new_tab('https://www.amazon.in/s?k=triphala+shilajit+and+methi')
	st.subheader('Vijaysar')
	img3 = Image.open('kino.jpg')
	st.image(img3)
	st.text('''Vijaysar also known as Indian Kino Tree or Pterocarpus Marsupium is the most appreciated\nherb for controlling blood sugar in diabetes. Aqueous Infusion prepared\nin Herbal Wood Glass is used for controlling blood sugar.''')
	if st.button('Buy now'):
		webbrowser.open_new_tab('https://www.amazon.in/s?k=triphala+shilajit+and+methi')
	st.subheader('Gurmar')
	img2 = Image.open('d3.png')
	st.image(img2)
	st.text('''Gymnema sylvestre, also known as Gurmar in India (Gur meaning sugar and\n mar meaning destroy) may help you fight sugar cravings and lower high blood sugar levels.\n The plant may also play a beneficial role in diabetes treatment, as it may \nhelp stimulate insulin secretion and the regeneration of pancreas islet \ncells — both of which can help lower blood sugar.''')
	if st.button('Shop now'):
		webbrowser.open_new_tab('https://www.amazon.in/s?k=gymnema+sylvestre')