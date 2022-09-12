import sqlite3
import xlsxwriter

import sys
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

def writeTofile(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)
    print("Stored blob data into: ", filename, "\n")

def readBlobData(empId):
    workbook = xlsxwriter.Workbook('keypoints.xlsx')
    worksheet = workbook.add_worksheet()
    try:
        sqliteConnection = sqlite3.connect('database.db')
        # creating connection
        # conn = sqlite3.connector.connect(
        #         # host="localhost",
        #         # user="sammy",
        #         # password="password",
        #         database ="database.db"
        #         )
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sql_fetch_blob_query = """SELECT data FROM keypoints """
        cursor.execute(sql_fetch_blob_query, (empId,))
        record = cursor.fetchall()
        for column in record:
            print("X = ", column[0], "Y = ", column[1])
            X = column[0]
            Y = column[2]
            #descriptor = column[3]

            print("Storing keypoint data on disk \n")
            keypointPath = "~/COMP0132/brighton_workspace/keypoints.csv"
            #writeTofile(photo, photoPath)
            row = 0
            for item in X:
                worksheet.write(row, 0, X)
                worksheet.write(row, 1, Y)
                row += 1
            

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")


def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

blob_to_array()
readBlobData(1)
