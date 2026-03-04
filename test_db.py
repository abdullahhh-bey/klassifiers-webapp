import pyodbc

servers = ["localhost", r"localhost\SQLEXPRESS", "(localdb)\\MSSQLLocalDB", "D", r"D\SQLEXPRESS", "DESKTOP-D642JM0"]
for server in servers:
    try:
        conn = pyodbc.connect(f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE=klassifier;Trusted_Connection=yes", timeout=2)
        print(f"Success with {server}")
        conn.close()
        break
    except Exception as e:
        print(f"Failed with {server}")
