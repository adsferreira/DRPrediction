# false rectangle which cleans the internal part of the squircle 
i_len_x = 0.79
i_len_y = 0.44

o_len_x = 0.868
o_len_y = 0.25

dis_x = o_len_x - i_len_x
dis_y = i_len_y - o_len_y

step_x = dis_x / 5
step_y = dis_y / 5

print(dis_x)
print(dis_y)

print("step in x", step_x)
print("step in y", step_y)
print("\n")

c_len_x = i_len_x
c_len_y = i_len_y

while c_len_x <= (o_len_x + 5 * step_x):
    print("current x length", round(c_len_x, 3))
    print("current y length", round(c_len_y, 3))
    c_len_x += step_x
    c_len_y -= step_y