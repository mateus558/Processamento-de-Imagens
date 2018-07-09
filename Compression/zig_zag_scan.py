import numpy as np

def zigzag(input, channels=3):
    w = 0
    h = 0

    height_min = 0
    width_min = 0

    height = input.shape[0]
    width = input.shape[1]

    i = 0

    block_out = np.zeros(( height * width * channels))

    while ((h < height) and (w < width)):
        
        if ((w + h) % 2) == 0:                 # going up

            if (h == height_min):
                block_out[i] = input[h, w, 0]; i += 1      # if we got to the first line
                block_out[i] = input[h, w, 1]; i += 1
                block_out[i] = input[h, w, 2]; i += 1

                if (w == width):
                    h += 1
                else:
                    w += 1

            elif ((w == width -1 ) and (h < height)):   # if we got to the last column
                block_out[i] = input[h, w, 0]; i += 1
                block_out[i] = input[h, w, 1]; i += 1
                block_out[i] = input[h, w, 2]; i += 1

                h += 1

            elif ((h > height_min) and (w < width -1 )):    # all other cases
                block_out[i] = input[h, w, 0]; i += 1
                block_out[i] = input[h, w, 1]; i += 1
                block_out[i] = input[h, w, 2]; i += 1

                h -= 1
                w += 1
        
        else:                                    # going down

            if ((h == height -1) and (w <= width -1)):       # if we got to the last line
                block_out[i] = input[h, w, 0]; i += 1
                block_out[i] = input[h, w, 1]; i += 1
                block_out[i] = input[h, w, 2]; i += 1

                w += 1
        
            elif (w == width_min):                  # if we got to the first column
                block_out[i] = input[h, w, 0]; i += 1
                block_out[i] = input[h, w, 1]; i += 1
                block_out[i] = input[h, w, 2]; i += 1

                if (h == height -1):
                    w += 1
                else:
                    h += 1

            elif ((h < height -1) and (w > width_min)):     # all other cases
                block_out[i] = input[h, w, 0]; i += 1
                block_out[i] = input[h, w, 1]; i += 1
                block_out[i] = input[h, w, 2]; i += 1

                h += 1
                w -= 1

        if ((h == height-1) and (w == width-1)):          # bottom right element
            block_out[i] = input[h, w, 0]; i += 1
            block_out[i] = input[h, w, 1]; i += 1
            block_out[i] = input[h, w, 2]; i += 1

            break

    return block_out


def inverse_zigzag(input, height, width, channels=3):
    w = 0
    h = 0

    height_min = 0
    width_min = 0

    block_out = np.zeros((height, width, channels))

    i = 0

    while ((h < height) and (w < width)): 

        if ((w + h) % 2) == 0:                   # going up
            
            if (h == height_min):                       # if we got to the first line
                block_out[h, w, 0] = input[i]; i += 1
                block_out[h, w, 1] = input[i]; i += 1
                block_out[h, w, 2] = input[i]; i += 1

                if (w == width):
                    h += 1
                else:
                    w += 1

            elif ((w == width -1 ) and (h < height)):   # if we got to the last column
                block_out[h, w, 0] = input[i]; i += 1
                block_out[h, w, 1] = input[i]; i += 1
                block_out[h, w, 2] = input[i]; i += 1

                h += 1

            elif ((h > height_min) and (w < width -1 )):    # all other cases
                block_out[h, w, 0] = input[i]; i += 1
                block_out[h, w, 1] = input[i]; i += 1
                block_out[h, w, 2] = input[i]; i += 1

                h -= 1
                w += 1
        
        else:                                    # going down

            if ((h == height -1) and (w <= width -1)):       # if we got to the last line
                block_out[h, w, 0] = input[i]; i += 1
                block_out[h, w, 1] = input[i]; i += 1
                block_out[h, w, 2] = input[i]; i += 1

                w += 1
        
            elif (w == width_min):                  # if we got to the first column
                block_out[h, w, 0] = input[i]; i += 1
                block_out[h, w, 1] = input[i]; i += 1
                block_out[h, w, 2] = input[i]; i += 1

                if (h == height -1):
                    w += 1
                else:
                    h += 1
                                
            elif((h < height -1) and (w > width_min)):     # all other cases
                block_out[h, w, 0] = input[i]; i += 1
                block_out[h, w, 1] = input[i]; i += 1
                block_out[h, w, 2] = input[i]; i += 1

                h += 1
                w -= 1




        if ((h == height-1) and (w == width-1)):          # bottom right element
            block_out[h, w, 0] = input[i]; i += 1
            block_out[h, w, 1] = input[i]; i += 1
            block_out[h, w, 2] = input[i]; i += 1

            break


    return block_out



