import serial
calibrate = 0
data = serial.Serial("com9 ", baudrate=9600)

while True :

    import cv2, numpy as np
    import  time


    # PHASE 1 ############################################################################################################################################################

    cap = cv2.VideoCapture(0)
    w = int(cap.get(3))
    h = int(cap.get(4))
    print(w, h)
    black = np.ones((h, w, 3), np.uint8)
    draw = np.ones((h, w, 3), np.uint8)
    # draw = np.full((h,w,3),255,dtype=np.uint8)
    minus_img = np.ones((h, w, 3), np.uint8)

    k = 0
    kk = 0
    t = 0

    pts = []
    timer = 0


    def time_up():
        global k, pts
        k = 0
        pts = []


    kernel = np.ones((5, 5), np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        frame1 = frame.copy()
        blur_in = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur_in, cv2.COLOR_BGR2HSV)




        g = np.uint8([[[0, 255, 0]]])
        green_l = np.array([50, 80, 100])
        green_h = np.array([70, 255, 255])

        mask = cv2.inRange(hsv, green_l, green_h)
        blur = cv2.medianBlur(mask, 5)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        adapth = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


        contours, _ = cv2.findContours(adapth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # print(len(contours),"kkkk")

        cv2.drawContours(frame1, contours, -1, (255, 0, 0), 3)

        for contour in contours:
            if 500 < cv2.contourArea(contour) < 1500:
                print("in")
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 3)
                pts.append((int(x), int(y)))

                t = 0
                if k > 0:
                    cv2.line(draw, pts[k - 1], pts[k], (255, 255, 255), 3,cv2.LINE_AA)  ###############################################################################################################
                    cv2.line(black, pts[k - 1], pts[k], (255, 255, 0), 3)
                    timer = round(time.time())
                    k += 1
                if k == 0:
                    k += 1
                print(k)
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)
                # cv2.circle(black, (int(x), int(y)),8, (255,255, 255), -1)
                break
            else:

                if round(time.time()) == (timer + 2) and t == 0:
                    print("time up", timer, time.time())
                    t = 1
                    k = 0
                    pts = []

        flip = cv2.flip(frame, 1)

        black_flip = cv2.flip(black, 1)
        fin = cv2.add(flip, black_flip)             #======================= imp

        cv2.line(minus_img, (0, 240), (640, 240), (0, 0, 255), 2)

        fin = cv2.addWeighted(fin, 0.7, minus_img, 0.3, 0)
        draw1 = cv2.flip(draw, 1)          #=========================== imp

        # print(green_h,green_l)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        # cv2.imshow("open",open)
        # cv2.imshow("erode", erode)
        cv2.imshow("blur", blur)
        cv2.imshow("adapt_thresh", adapth)
        cv2.imshow("flip", flip)
        cv2.imshow("bflip", black_flip)
        cv2.imshow("final", fin)
        cv2.imshow("finak", draw1)

        # cv2.imshow("dilate", dilate)
        # cv2.imshow("frame1", frame1)

        if cv2.waitKey(1) & 0xFF == ord('b'):
            break

    cap.release()

    print(pts)

    cv2.destroyAllWindows()

    cv2.imwrite("digit.png", draw1)


    #  PHASE 2 ########################################################################################################3333333333##################


    import cv2, numpy as np
    import torch
    import torchvision
    from PIL import Image
    import torch.nn.functional as F  # for prediction
    from torchvision import datasets, transforms
    from torch import nn

    # for prediction of digit

    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # .ToTensor() converts numpy array (0 - 255) to float Tensor (0 - 1)


    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()

        return model


    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.dropout1 = nn.Dropout(0.5)  # 0.5 ---- fraction of nodes to turn of
            self.fc2 = nn.Linear(500, 10)

        def forward(self, X):
            x = F.relu(self.conv1(X))
            x = F.max_pool2d(x, 2, 2)  # 2,2 -- kernal size
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)  # 2,2 -- kernal size
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.fc2(x)  # raw output called as score -- to avoid to pass in values very close to zero or 1 (in loss function)

            return x


    model = load_checkpoint('MNIST_CNN.pth')
    print(model)


    def digit_predict(path):
        imge = cv2.imread(path)

        # You may need to convert the color.
        # imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(imge)
        image = image.convert('1')

        image = transform(image)
        print(image[0].unsqueeze(0).shape, "*" * 100)
        print(image[0].unsqueeze(0).unsqueeze(0).shape, "*" * 100)
        image = image[0].unsqueeze(0).unsqueeze(0)  # to change it to 4 dimensions

        output = model(image)
        _, pred = torch.max(output, 1)
        print("prediction : ",pred.item())
        return pred.item()


    # the image to sort the contours


    img = cv2.imread("digit.png", 0)
    img1 = cv2.imread("digit.png")
    img2 = cv2.imread("digit.png")

    kernel = np.ones((5, 5), np.uint8)

    img5 = cv2.imread("digit.png")

    # print(w, h, "*", frame1.shape )

    h_h = img.shape[1]
    w_w = img.shape[0]
    print(w_w, h_h)
    x_img = np.ones((w_w, h_h, 3), np.uint8)
    y_img = np.ones((w_w, h_h, 3), np.uint8)
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = []
    area_string = []
    x_points_str = []
    y_points_str = []
    x_points = []
    y_points = []
    y_h = []
    height = []
    height_str = []
    k = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        cx = (x + (w // 2))
        cy = (y + (h // 2))

        area.append(w * h)
        area_string.append(f"{w * h}_{k}")   #k is index
        height.append(h)
        height_str.append(f"{h}_{k}")
        x_points.append(x)
        y_points.append(y)
        y_h.append(y + h)
        x_points_str.append(f"{x}_{k}")
        y_points_str.append(f"{y}_{k}")

        k += 1

        # (x1, y1), (MA, ma), angle = cv2.fitEllipse(cnt)

        (xc, yc), radius = cv2.minEnclosingCircle(cnt)
        #center = (int(xc), int(yc))
        cv2.rectangle(img1, (x, y), (x + w, y + h), (20, 255, 0), 3)
        # cv2.circle(img1, center, int(radius), (0,255,255), 2)

        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img1, [box], 0, (0, 0, 255), 2)

    x_points_sorted = []
    y_points_sorted = []
    area_sorted = []
    x_points_copy = x_points.copy()

    val = max(x_points)

    for i in range(len(x_points)):
        minimum = min(x_points_copy)
        index = x_points.index(minimum)

        x_points_sorted.append(x_points_str[index])
        y_points_sorted.append(y_points_str[index])
        area_sorted.append(area_string[index])

        x_points_copy[index] = val + 3

    # finding comma from max y at the bottom y+h or the bounding rect
    comma_index = y_h.index(max(y_h))

    # finding  the x and y from the sorted array

    lol = 0
    k = 0
    init_x = 0
    init_y = 0
    final_x_pts = ""
    final_y_pts = ""
    for i in x_points_sorted:

        if i[-1] == str(comma_index):
            lol = 1
            continue
        x, y, w, h = cv2.boundingRect(cnts[int(i[-1])])
        # ROI = img2[y-30:y+h+30, x-30:x+w+30]
        if lol == 0:
            if (y > 240) and (init_x == 0):
                init_x = 1
                final_x_pts += "-"
                print("its a minus bro")
            else:
                # x_img[y:y+h, x:x+w]  = ROI
                # cv2.imwrite("x_ROI.png", ROI)
                cv2.drawContours(x_img, cnts[int(i[-1])], -1, (255, 255, 255), 6)
                ROI_x = x_img[(y - 40):(y + h + 40), (x - 40):(x + w + 40)]
                ROI_x = cv2.dilate(ROI_x, kernel, iterations=2)
                cv2.imwrite("x_ROI_coor.png", ROI_x)
                cv2.imshow("ROI_x", ROI_x)
                cv2.waitKey(0)
                final_x_pts += str(digit_predict("x_ROI_coor.png"))
                x_img = np.ones((w_w, h_h, 3), np.uint8)



        else:
            if (y > 240) and (init_y == 0):
                init_y = 1
                final_y_pts += "-"
                print("its a minus bro")
            else:
                # y_img[y:y+h, x:x+w] = ROI
                # cv2.imwrite("y_ROI.png", ROI)
                cv2.drawContours(y_img, cnts[int(i[-1])], -1, (255, 255, 255), 6)
                ROI_y = y_img[(y - 40):(y + h + 40), (x - 40):(x + w + 40)]
                ROI_y = cv2.dilate(ROI_y, kernel, iterations=2)
                cv2.imwrite("y_ROI_coor.png", ROI_y)
                cv2.imshow("ROI_y", ROI_y)
                cv2.waitKey(0)
                final_y_pts += str(digit_predict("y_ROI_coor.png"))
                y_img = np.ones((w_w, h_h, 3), np.uint8)

    print(area)
    print(area_string)
    print(area_sorted)
    print(y_points)
    print(y_points_sorted)
    print(x_points)
    print(x_points_sorted)
    print(x_points_copy)
    print(height, height_str)

    cv2.line(img1, (0, 240), (640, 240), (0, 0, 255), 2)

    cv2.imshow("img1", img1)
    cv2.imshow("img", img)
    cv2.imshow("x", x_img)
    cv2.imshow("y", y_img)
    # cv2.imshow("mask",mask)

    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("x_coordinate.png", x_img)
    cv2.imwrite("y_coordinate.png", y_img)

    print("*" * 200)
    final_coordinates = [int(final_x_pts), int(final_y_pts)]

    print("x and y coordinates are : ", final_coordinates)

     #  PHASE 3 #################################################################################################################################################

    wave = int(input("enter 1 to send the coordinates to the bot else enter 0 to again do image processing : "))

    if wave == 1:


        x = int(final_coordinates[0])
        y = int(final_coordinates[1])

        data.flushInput()  # to flush away the previous inputs recieved from arduino

        if calibrate == 0:
            print("calibrating...........................")
            calibrate = 1
            data.write(str(3).encode()) #somevalue
            curr_heading =  int(data.readline().decode())


            print("calibration done !")
            time.sleep(1)

        val = int(input("entry some value to proceed"))

        data.write(str(y).encode())
        print("entered y coordinate : ", data.readline().decode())

        time.sleep(2)

        data.write(str(x).encode())
        print("entered x coordinate : ", data.readline().decode())

        while True:
            done = data.readline().decode()
            print(done)
            if("z" in done):
                print(done.split())
                break


        enter_val = int(input("enter 1 to continue the same process or enter 0 if you are done : "))
        if enter_val == 1:
            continue
        else:
            break
    else :
        pass











