import os
import cv2
import face_recognition

FACES_DIR = "faces"

# Capturar imágenes para una nueva persona
def capture_images():
    name = input("Nombre de la persona: ").strip()
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    video_capture = cv2.VideoCapture(0)
    count = 0

    print("Presiona 'c' para capturar, 'q' para salir.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        cv2.imshow('Captura', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Guardada: {img_path}")
            count += 1
        elif key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Cargar caras conocidas
def load_faces():
    known_face_encodings = []
    known_face_names = []

    print("Cargando imágenes conocidas...")
    for person_name in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
            else:
                print(f"No se encontró cara en {img_path}, omitiendo.")

    print(f"Cargadas {len(known_face_encodings)} caras conocidas.")
    return known_face_encodings, known_face_names


# Reconocimiento facial en vivo
def recognize_faces(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No se pudo capturar la imagen de la cámara.")
            break

        # Reducir tamaño para acelerar
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detectar ubicaciones de caras
        face_locations = face_recognition.face_locations(rgb_small, model="hog")

        # Evitar el error: validar coordenadas antes de codificar
        if face_locations and all(isinstance(loc, tuple) and len(loc) == 4 for loc in face_locations):
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        else:
            face_encodings = []

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        # Dibujar resultados
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Menú principal
if __name__ == "__main__":
    while True:
        print("\n1. Capturar imágenes")
        print("2. Iniciar reconocimiento")
        print("3. Salir")
        option = input("Opción: ").strip()

        if option == "1":
            capture_images()
        elif option == "2":
            faces, names = load_faces()
            recognize_faces(faces, names)
        elif option == "3":
            break
        else:
            print("Opción no válida.")
