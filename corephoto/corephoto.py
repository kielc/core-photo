import os
import re

import cv2
import lasio
import matplotlib.pyplot as plt
import numpy as np


class CorePhoto:
    def __init__(
        self, photo_path, image_extension=".tif", regex=r"[0-9]{3,4}\.[0-9]{2}"
    ):
        r"""CorePhoto Class


        Parameters
        ----------
        photo_path : raw string
            Full path to directory containing core photos.
            e.g. r"C:\Users\username\Documents\corephotos"
        image_extension : string, optional
            File extension for the type of images being read. The default is ".tif".
        regex : raw string, optional
            Regular expression applied to the image file names using re.findall() to
            generate a two element list containing first, the top depth, and second,
            the bottom depth. The default is r"[0-9]{3,4}\.[0-9]{2}".

        Returns
        -------
        None

        """
        self.photo_path = photo_path
        self.image_extension = image_extension
        self.regex = regex
        self._dir_iter = os.scandir(photo_path)
        self.core_images = []
        self.top = None
        self.bottom = None

    def _morph_image(self, img):
        """Converts an image to HSV colorspace, applies an HSV filtering mask, performs
        a closing morphological transformation, and returns the resulting image.

        The trasformations are required so that contours of the core segements can be
        selected from a core photo image.

        """
        # hsv opencv range 0-179, 0-255, 0-255
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([179, 100, 100])
        kernel = np.ones((7, 7), np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return morph

    def _find_segment_contours(self, morph):
        """Finds and returns a list of contours of the core segments from a
        morphologically transformed core photo image.

        """

        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # get 6 largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
        # get 4 smallest of the 6 largest contours (4 core segments)
        contours = sorted(contours, key=cv2.contourArea, reverse=False)[:4]

        return contours

    def _crop_segment_images(self, contours, img):
        """Returns a list of core segment images cropped from a core photo image by
        creating bounding boxes from a list of contours.

        """
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # sort bounding boxes top to bottom
        # b is a tuple of two tuples (contours, bounding_box),
        # b[1][1] is y position in bounding_box
        contours, bounding_boxes = zip(
            *sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1])
        )

        segment_images = []
        for box in bounding_boxes:
            x, y, w, h = box
            # exclude top/bottom 10% of make sure only rock is cropped
            x1, y1 = x, round(y + h * 0.10)
            x2, y2 = x + w, round(y + h * 0.90)
            segment_images.append(img[y1:y2, x1:x2])

        return segment_images

    def _extract_core_image(self, img):
        """Returns an image of concatenated core segments cropped from a core photo
        image.

        """
        morph = self._morph_image(img)
        contours = self._find_segment_contours(morph)
        segment_images = self._crop_segment_images(contours, img)
        core_image = self._concat_core_segments(segment_images)

        return core_image

    def _concat_core_segments(self, img_list):
        """Returns an image of concatenated core segments from a list of individual
        core segment images.

        """
        new_height = min(img_list, key=lambda x: x.shape[0]).shape[0]

        img_list_cropped = [self._center_crop_vert(img, new_height) for img in img_list]

        return np.hstack(img_list_cropped)

    def _center_crop_vert(self, img, new_height):
        """Crops the vertical dimension of an image to a new_height (pixels) from the
        center.

        """
        height = img.shape[0]

        top = int(np.ceil((height - new_height) / 2))
        bottom = int(height - np.floor((height - new_height) / 2))

        return img[top:bottom, ...]

    def process_photos(self):
        """Method run after instantiation of the class to process the core photos.


        Returns
        -------
        None

        """
        for i, entry in enumerate(self._dir_iter):
            if entry.is_file() and entry.name.endswith(self.image_extension):
                photo = cv2.imread(entry.path)
                depths = [float(s) for s in re.findall(self.regex, entry.name)]
                core_image = self._extract_core_image(photo)

                self.core_images.append(
                    {"image": core_image, "top": min(depths), "bottom": max(depths)}
                )

                print(f"{i+1} photo(s) processed into concatenated core image(s).")

        self.top = min(i["top"] for i in self.core_images)
        self.bottom = max(i["bottom"] for i in self.core_images)

    def core_interval_img(self, top=None, bottom=None):
        """
        loops over each photo in dict to see if top/bottom in the img depth
        concatenates the images
        crops from top to bottom
        returns the image


        Parameters
        ----------
        top : float, optional
            Top depth. The default is None.
        bottom : float, optional
            Bottom depth. The default is None.

        Returns
        -------
        interval_cropped : cv2 image
            Image of core cropped from top to bottom.

        """
        top = top or self.top
        bottom = bottom or self.bottom

        if top < self.top or bottom > self.bottom:
            raise Exception("Interval outside of core depth.")

        selected_imgs = [
            img_dict
            for img_dict in self.core_images
            if range_overlap((img_dict["top"], img_dict["bottom"]), (top, bottom))
        ]

        top_selected = min([img_dict["top"] for img_dict in selected_imgs])
        bottom_selected = max([img_dict["bottom"] for img_dict in selected_imgs])
        length_selected = bottom_selected - top_selected

        selected_concat = self._concat_core_segments(
            [
                img_dict["image"]
                for img_dict in sorted(selected_imgs, key=lambda d: d["top"])
            ]
        )

        top_px = round(
            ((top - top_selected) / length_selected) * selected_concat.shape[1]
        )
        bottom_px = round(
            ((bottom - top_selected) / length_selected) * selected_concat.shape[1]
        )

        interval_cropped = selected_concat[:, top_px:bottom_px, :]

        """        
        # remove print statements         
        print("top", top)
        print("bottom", bottom)
        print("top_photo_selected", top_selected)
        print("bottom_photo_selected", bottom_selected)
        print("length_photo_selected", length_selected)
        print("number_photos_in_range", len(selected_imgs))
        print("top at % from top_photo_selected", (top - top_selected) / length_selected)
        print("bottom at % from top_photo_selected", (bottom - top_selected) / length_selected)
        print("top_px", top_px)
        print("bottom_px", bottom_px)
        """

        return interval_cropped

    def edges_img(self, img):
        """Returns resulting image from canny edge detection.


        Parameters
        ----------
        img : cv2 image

        Returns
        -------
        edges_img : cv2 image

        """
        edges_img = cv2.Canny(img, 100, 200)

        return edges_img

    def edges_log(self, edges_img):
        """Returns numpy array of edges log.


        Parameters
        ----------
        edges_img : cv2 image
            Resulting image from canny edge detection returned by edges_img method.

        Returns
        -------
        edges_log : (N,) numpy.ndarray

        """
        edges_log = np.array([column.mean() / 255 for column in edges_img.T])

        return edges_log

    def grayscale_log(self, img):
        """Returns numpy array of grayscale color log.


        Parameters
        ----------
        img : cv2 image
            Resulting image returned by core_interval_img method.

        Returns
        -------
        gray_log : (N,) numpy.ndarray

        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_log = np.array([column.mean() / 255 for column in gray_img.T])

        return gray_log

    def core_interval_display(self, top=None, bottom=None):
        """Displays core log with 4 tracks.
        -Core Image
        -Edge Detection Image
        -Edge Log
        -Grayscale Log


        Parameters
        ----------
        top : float, optional
            Top depth. The default is None.
        bottom : float, optional
            Bottom depth. The default is None.

        Returns
        -------
        None

        """
        top = top or self.top
        bottom = bottom or self.bottom

        img = self.core_interval_img(top, bottom)
        depth = np.linspace(top, bottom, img.shape[1])

        fig, ax = plt.subplots(
            ncols=4, figsize=(10, 10), gridspec_kw={"width_ratios": [1, 1, 2, 2]}
        )

        # core image subplot
        ax[0].imshow(opencv2matplotlib(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)))
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].set_xlabel("Core Image")

        # edge detection image subplot
        ax[1].imshow(cv2.rotate(self.edges_img(img), cv2.ROTATE_90_CLOCKWISE))
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].set_xlabel("Edge Detection Image")

        # edge log subplot
        ax[2].plot(self.edges_log(self.edges_img(img)), depth)
        ax[2].set_xlim(0, 1)
        ax[2].set_ylim(top, bottom)
        ax[2].invert_yaxis()
        ax[2].set_xlabel("Edge Log")

        # grayscale log subplot
        ax[3].plot(self.grayscale_log(img), depth)
        ax[3].set_xlim(0, 1)
        ax[3].set_ylim(top, bottom)
        ax[3].invert_yaxis()
        ax[3].set_xlabel("Grayscale Log")

        plt.tight_layout()

        """        
        # remove print statements       
        print(min(depth))
        print(max(depth))
        print(depth.shape)
        print("img.shape:", img.shape)
        print("self.edges_img(img).shape:", self.edges_img(img).shape)
        print("self.edges_log(self.edges_img(img)).shape:", self.edges_log(self.edges_img(img)).shape)
        print("self.grayscale_log(img).shape:", self.grayscale_log(img).shape)
        """

    def write_las(self, header_dictionary, filename, top=None, bottom=None):
        """Writes a las file containing a header and 3 log curves.
        -depth
        -edges log
        -grayscale log


        Parameters
        ----------
        header_dictionary : dict
            Header section metadata.
        filename : str
            Name of file to be written to disk.
        top : float, optional
            Top depth. The default is None.
        bottom : float, optional
            Bottom depth. The default is None.

        Returns
        -------
        None

        """
        top = top or self.top
        bottom = bottom or self.bottom

        img = self.core_interval_img(top, bottom)

        las = lasio.LASFile()

        for key, value in header_dictionary.items():
            setattr(las.well, key, value)

        las.add_curve("DEPTH", np.linspace(top, bottom, img.shape[1]), unit="m")
        las.add_curve("EDGES", self.edges_log(self.edges_img(img)))
        las.add_curve("GRAYSCALE", self.grayscale_log(img))
        las.write(filename)


def opencv2matplotlib(image):
    """Converts image from BGR to RGB color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def range_overlap(range_x, range_y):
    """Tests if two ranges overlap.


    Parameters
    ----------
    range_x : tuple[float, float]
        (range_bottom, range_top)
    range_y : tuple[float, float]
        (range_bottom, range_top)

    Returns
    -------
    bool

    """
    if range_x[0] > range_x[1] or range_y[0] > range_y[1]:
        raise Exception("range[0] must be less than or equal to range[1]")
    return range_x[0] <= range_y[1] and range_y[0] <= range_x[1]
