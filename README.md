# SVNCKH điểm danh nhận diện khuôn mặt (Chưa hoàn thiện)
## Installation Requirement

``bash
pip install -r .\requirements.txt
``
## Get Face data
MTCNN sẽ xác định vị trí các khuôn mặt trong ảnh và đưa ra 1 box chưa các tọa độ mặt
``python
mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)
``
MTCNN sẽ trả về ảnh 160 * 160 pixel, thông thường sẽ là ảnh sát mặt nhưng khi thêm margin vào thì vùng ảnh đấy sẽ được mở rộng ra theo số pixel đã chọn (20)
Ta thêm margin vào để có thể dự đoán ảnh mặt tốt hơn thứ sẽ nói ở phần sau.
Khi lấy được ảnh mặt thì ta sẽ embed nó sử dụng model của facenet
``python
model = InceptionResnetV1(
	classify=False,
	pretrained="casia-webface"
).to(device)
model.eval()
``
Tập model này đã được pretrain qua tập data "casia-webface", tuy nhiên những khuôn mặt trong tập data này hầu hết của của người châu âu nên đối với mặt người châu Á như chúng ta sẽ có chút sai số.
Ảnh qua facenet sẽ trả về 1 embed vector (có thể gọi là vector đặc trưng), trong bài này vector sẽ có 512 chiều.
![image](https://user-images.githubusercontent.com/78363603/174589201-91ee684a-9332-4210-b84e-25f008348d5d.png)

Trong không gian vector đó chúng ta sẽ tạo 1 label chứa vector trung bình của những vector thuộc cùng 1 mặt người.

![image](https://user-images.githubusercontent.com/78363603/174600303-68f97f35-43a8-40a7-affb-a44282c894f2.png)

Hình tròn đỏ là trung bình của những hình tròn, hình vuông xanh cũng thế
## Face Recognition
Như đã nói ở phần trước thì MTCNN có phần margin=20.
Lí giải tại sao lại có margin là MTCNN sẽ luôn trả về box vuông và chứa các tọa độ khuôn mặt
Tuy nhiên những mặt ở góc ảnh sẽ trả về những giá trị tọa độ âm=> chúng ta phải căn chỉnh lại và dùng margin
Ý tưởng chính trong bài là giãn ảnh mặt ra và giới hạn bởi các tọa độ khung ảnh.

Khi đã lấy ảnh mặt rồi thì ta sẽ làm như bước "Get Face data" để lấy embed vector
Vì thuật toán vẫn còn sơ khai và chưa hoàn thiện nên trong bài mình chỉ so sánh vector của mặt đầu vào với những embed vector trung bình có sẵn

![image](https://user-images.githubusercontent.com/78363603/174601448-5b30200d-517e-409b-8320-ef9ced413700.png)

Khoảng cách đến điểm vuông xanh nhỏ hơn=> nó thuộc class vuông xanh

![image](https://user-images.githubusercontent.com/78363603/174601554-75d84b96-48f0-48c7-9643-7c6fc12c61ce.png)

