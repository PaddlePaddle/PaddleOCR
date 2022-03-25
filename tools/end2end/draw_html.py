import os


def draw_debug_img(html_path):

    err_cnt = 0
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write(
            "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
        )
        image_list = []
        path = "./det_results/310_gt/"
        #path = "infer_results/"
        for i, filename in enumerate(sorted(os.listdir(path))):
            if filename.endswith("txt"): continue
            print(filename)
            # The image path
            base = "{}/{}".format(path, filename)
            base_2 = "../PaddleOCR/det_results/ch_PPOCRV2_infer/{}".format(
                filename)
            base_3 = "../PaddleOCR/det_results/ch_ppocr_mobile_infer/{}".format(
                filename)

            if True:
                html.write("<tr>\n")
                html.write(f'<td> {filename}\n GT')
                html.write('<td>GT\n<img src="%s" width=640></td>' % (base))
                html.write('<td>PPOCRV2\n<img src="%s" width=640></td>' %
                           (base_2))
                html.write('<td>ppocr_mobile\n<img src="%s" width=640></td>' %
                           (base_3))

                html.write("</tr>\n")
        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    print("ok")
    #print("all cnt: {}, err cnt: {}, acc: {}".format(len(imgs), err_cnt, 1.0 * (len(imgs) - err_cnt) / len(imgs)))
    return


if __name__ == "__main__":

    html_path = "sys_visual_iou_310.html"

    draw_debug_img()
