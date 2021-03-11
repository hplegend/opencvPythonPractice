#! /usr/bin/env python
# coding: utf-8


in_file = open('./images/cv_points_xyz.txt', 'r')
out_file = open('./images/remove.txt', 'w')

inline = in_file.readline()
while '' != inline:
	if 'inf' not in inline:
		out_file.write(inline)

	inline = in_file.readline()

out_file.close()
in_file.close()
