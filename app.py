import streamlit as st
import pandas as pd
import sqlite3
from pymongo import MongoClient
from pathlib import Path
import re
import random

selected_database = 'local_database'
operator_mapping = {
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "greater than": ">",
    "less than": "<",
    "equal to": "=",
    "equals": "=",
    "not equal to": "<>",
    "not equal": "<>",
    "is": "==",
    "is not": "<>",
    "isn't": "<>"
}
aggregation_mapping = {
    "average": "AVG",
    "maximum": "MAX",
    "minimum": "MIN",
    "sum": "SUM",
    "total": "SUM",
    "count": "COUNT"
}
if "database_info" not in st.session_state:
    st.session_state["database_info"] = {"local_database": {}}
database_info = st.session_state["database_info"]
def get_sqlite_data(db_path, sql_query):
    db_path = db_path + '.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    data = cursor.fetchall()
    conn.close()
    return data
def get_mongo_data(db_path, sql_query):
    collection_name, pipeline = parse_sql_to_mongo(sql_query, database_info)
    print(pipeline)
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_path]
    collection = db[collection_name]
    results = collection.aggregate(pipeline)
    return pd.DataFrame(list(results))
def process_command(command, database_info):
    if command.startswith('\\'):
        tokens = command.strip().split()
        cmd = tokens[0][1:]  
        query = ""
        for db_name, db_content in database_info.items():
            db = db_content  

            if cmd == 'explore':
                queries = []
                for table_name, table_info in db.items():
                    columns = list(table_info['columns'].keys())
                    columns_str = ', '.join(columns)
                    query = f"SELECT {columns_str} \nFROM {table_name} \nLIMIT 3;"
                    print(f"Database: {db_name}\nGenerated Query for '{table_name}':\n{query}\n")
                    queries.append(query)
                return queries

            elif cmd == 'sample':
                table_name = random.choice(list(db.keys()))
                table_info = db[table_name]
                columns = list(table_info['columns'].keys())
                column_name = random.choice(columns)
                column_type = table_info['columns'][column_name]
                sample_values = table_info['samples'][column_name]
                if sample_values:
                    condition_value = random.choice(sample_values)
                    if column_type == 'TEXT':
                        condition_value = f"'{condition_value}'"
                else:
                    if column_type == 'INTEGER':
                        condition_value = random.randint(1, 100)
                    elif column_type == 'REAL':
                        condition_value = round(random.uniform(0.0, 4.0), 2)
                    elif column_type == 'TEXT':
                        condition_value = "'SampleValue'"
                    else:
                        condition_value = 'NULL'
                condition = f"{column_name} = {condition_value}"
                columns_str = ', '.join(columns)
                query = f"SELECT {columns_str} \nFROM {table_name} \nWHERE {condition};"
                print(f"Database: {db_name}\nGenerated Sample Query:\n{query}\n")

            elif cmd == 'groupby':
                table_name = random.choice(list(db.keys()))
                table_info = db[table_name]
                columns = list(table_info['columns'].keys())
                if len(columns) < 2:
                    print(f"Database: {db_name}\nNot enough columns in table '{table_name}' for GROUP BY.\n")
                    continue
                column1 = random.choice(columns)
                columns.remove(column1)
                column2 = random.choice(columns)
                aggregation_function = random.choice(['SUM', 'AVG', 'COUNT', 'MAX', 'MIN'])
                columns_str = ', '.join([column1, f"{aggregation_function}({column2}) AS {aggregation_function}_{column2}"])
                query = f"SELECT {columns_str}\nFROM {table_name}\nGROUP BY {column1};"
                print(f"Database: {db_name}\nGenerated GROUP BY Query:\n{query}\n")

            elif cmd == 'join':
                if len(db) < 2:
                    print(f"Database: {db_name}\nNot enough tables to perform a JOIN.\n")
                    cmd = 'where' 
                table_names = list(db.keys())

                common_columns = set()
                try_count = 0
                while not common_columns:
                    table1, table2 = random.sample(table_names, 2)
                    table1_info = db[table1]
                    table2_info = db[table2]
                    common_columns = set(table1_info['columns'].keys()) & set(table2_info['columns'].keys())
                    try_count += 1
                    if try_count > 5:
                        print(f"Database: {db_name}\nCould not find common columns to join between '{table1}' and '{table2}'.\n")
                        cmd = 'where'  
                        break
                if not common_columns:
                    possible_keys = set(table1_info['columns'].keys()) & set(['StudentID', 'CourseID', 'EnrollmentID'])
                    possible_keys &= set(table2_info['columns'].keys())
                    common_columns = possible_keys
                if not common_columns:
                    print(f"Database: {db_name}\nNo common columns to join between '{table1}' and '{table2}'.\n")
                    continue
                join_column = random.choice(list(common_columns))

                table1_columns = [f"{table1}.{col}" for col in table1_info['columns'].keys()]
                table2_columns = [f"{table2}.{col}" for col in table2_info['columns'].keys()]
                columns_str = ', '.join(table1_columns + table2_columns)

                query = (
                    f"SELECT {columns_str}\n"
                    f"FROM {table1}\n"
                    f"JOIN {table2} ON {table1}.{join_column} = {table2}.{join_column};"
                )
                print(f"Database: {db_name}\nGenerated JOIN Query:\n{query}\n")

            elif cmd == 'where':
                table_name = random.choice(list(db.keys()))
                table_info = db[table_name]
                columns = list(table_info['columns'].keys())
                column_name = random.choice(columns)
                column_type = table_info['columns'][column_name]
                # Use sample values for condition
                sample_values = table_info.get('samples', {}).get(column_name, [])
                if sample_values:
                    condition_value = random.choice(sample_values)
                    if column_type == 'TEXT':
                        condition_value = f"'{condition_value}'"
                    condition = f"{column_name} = {condition_value}"
                else:
                    if column_type == 'INTEGER':
                        condition_value = random.randint(1, 100)
                        condition = f"{column_name} > {condition_value}"
                    elif column_type == 'REAL':
                        condition_value = round(random.uniform(0.0, 100.0), 2)
                        condition = f"{column_name} < {condition_value}"
                    elif column_type == 'TEXT':
                        condition_value = "'SampleValue'"
                        condition = f"{column_name} = {condition_value}"
                    else:
                        condition = f"{column_name} IS NOT NULL"
                columns_str = ', '.join(columns)
                query = f"SELECT {columns_str}\nFROM {table_name}\nWHERE {condition};"
                print(f"Database: {db_name}\nGenerated WHERE Query:\n{query}\n")

            elif cmd == 'orderby':
                table_name = random.choice(list(db.keys()))
                table_info = db[table_name]
                columns = list(table_info['columns'].keys())
                column_name = random.choice(columns)
                order = random.choice(['ASC', 'DESC'])
                columns_str = ', '.join(columns)
                query = f"SELECT {columns_str}\nFROM {table_name}\nORDER BY {column_name} {order};"
                print(f"Database: {db_name}\nGenerated ORDER BY Query:\n{query}\n")

            elif cmd == 'limit':
                table_name = random.choice(list(db.keys()))
                table_info = db[table_name]
                columns = list(table_info['columns'].keys())
                limit_value = random.randint(1, 10)
                columns_str = ', '.join(columns)
                query = f"SELECT {columns_str}\nFROM {table_name}\nLIMIT {limit_value};"
                print(f"Database: {db_name}\nGenerated LIMIT Query:\n{query}\n")

            elif cmd == 'having':
                # For HAVING, we need GROUP BY and an aggregate function
                table_name = random.choice(list(db.keys()))
                table_info = db[table_name]
                columns = list(table_info['columns'].keys())
                if len(columns) < 2:
                    print(f"Database: {db_name}\nNot enough columns in table '{table_name}' for HAVING.\n")
                    continue
                column1 = random.choice(columns)
                columns.remove(column1)
                column2 = random.choice(columns)
                aggregation_function = random.choice(['SUM', 'AVG', 'COUNT', 'MAX', 'MIN'])
                condition_value = random.randint(1, 10)
                select_columns = ', '.join([column1, f"{aggregation_function}({column2}) AS {aggregation_function}_{column2}"])
                query = (
                    f"SELECT {select_columns}\n"
                    f"FROM {table_name}\n"
                    f"GROUP BY {column1}\n"
                    f"HAVING {aggregation_function}_{column2} > {condition_value};"
                )
                print(f"Database: {db_name}\nGenerated HAVING Query:\n{query}\n")
            
            else:
                print(f"Unknown command: {cmd}")
                break  
            break  
        return [query]
    else:
        query = parse_nl_query(command, database_info)
        return [query]
def tokenize_nl_query(nl_query):
    nl_query = re.sub(r'[^\w\s]', '', nl_query)
    tokens = nl_query.lower().split()
    return tokens
def format_value(value):
    try:
        float(value)
        return value 
    except ValueError:
        return f"'{value}'" 
def parse_condition(condition_tokens, aggregation_function=None, aggregation_field=None):
    condition_str = ' '.join(condition_tokens)

    for nl_op in sorted(operator_mapping.keys(), key=lambda x: -len(x)):
        if nl_op in condition_str:
            sql_op = operator_mapping[nl_op]
            condition_str = condition_str.replace(nl_op, sql_op)
            break  

    if aggregation_function and aggregation_function.lower() in condition_str:
        condition_str = condition_str.replace(aggregation_function.lower(), f"{aggregation_function}({aggregation_field})")

    match = re.match(r'([\w()]+)\s*([><=!]+)\s*(.+)', condition_str)
    if match:
        field = match.group(1)
        operator = match.group(2)
        value = format_value(match.group(3).strip())
        return f"{field} {operator} {value}"
    else:
        return condition_str 
def find_common_columns(table1, table2, database_info):
    columns1 = set(database_info[selected_database][table1]["columns"].keys())
    columns2 = set(database_info[selected_database][table2]["columns"].keys())
    common_columns = columns1.intersection(columns2)
    return list(common_columns)
def parse_nl_query(nl_query, database_info):
    tokens = tokenize_nl_query(nl_query)
    token_index = 0

    # Step 1: Check if first word is 'Calculate' or not
    is_aggregation = False
    if tokens[token_index] == 'calculate':
        is_aggregation = True
        token_index += 1
    elif tokens[token_index] in ['show', 'find']:
        token_index += 1
    else:
        return "Error: Query should start with 'Show', 'Find', or 'Calculate'."

    # Step 2: Check if next word is a number (Limit clause)
    limit_number = None
    if token_index < len(tokens) and tokens[token_index].isdigit():
        limit_number = tokens[token_index]
        token_index += 1

    # Step 3: Get all fields or aggregation of field
    fields = []
    aggregation_function = None
    if is_aggregation:
        if token_index < len(tokens):
            agg_word = tokens[token_index]
            if agg_word in aggregation_mapping:
                aggregation_function = aggregation_mapping[agg_word]
                token_index += 1
            else:
                return f"Error: Unknown aggregation function '{agg_word}'."

            if token_index < len(tokens) and tokens[token_index] == 'of':
                token_index += 1
            else:
                return "Error: Expected 'of' after aggregation function."

            if token_index < len(tokens):
                field = tokens[token_index]
                fields.append(field)
                token_index += 1
            else:
                return "Error: Expected field after 'of'."
        else:
            return "Error: Incomplete aggregation function."
    else:
        while token_index < len(tokens) and tokens[token_index] not in ['in', 'of']:
            field = tokens[token_index].strip(',')
            fields.append(field)
            token_index += 1

    # Step 4: Get all the tables
    if token_index < len(tokens) and tokens[token_index] in ['in', 'of']:
        token_index += 1
    else:
        return "Error: Expected 'in' or 'of' before table names."

    tables = []
    while token_index < len(tokens) and tokens[token_index] not in ['by', 'with', 'sorted', 'order']:
        table = tokens[token_index].strip(',')
        tables.append(table)
        token_index += 1

    if len(tables) == 0:
        return "Error: No table specified."

    # Handle JOIN if two tables
    join_condition = None
    if len(tables) == 2:
        common_columns = find_common_columns(tables[0], tables[1], database_info)
        if common_columns:
            # Use the first common column
            common_column = common_columns[0]
            join_condition = f"{tables[0]}.{common_column} = {tables[1]}.{common_column}"
        else:
            return "Error: No common columns found for JOIN between tables."

    # Step 5: Check for 'by' (GROUP BY)
    group_by_field = None
    if token_index < len(tokens) and tokens[token_index] == 'by':
        token_index += 1
        if token_index < len(tokens):
            group_by_field = tokens[token_index]
            token_index += 1
        else:
            return "Error: Expected field after 'by'."

    # Step 6: Check for 'with' (Condition)
    condition = None
    if token_index < len(tokens) and tokens[token_index] == 'with':
        token_index += 1
        condition_tokens = []
        while token_index < len(tokens) and tokens[token_index] not in ['sorted', 'order']:
            condition_tokens.append(tokens[token_index])
            token_index += 1
        condition = parse_condition(condition_tokens, aggregation_function=aggregation_function, aggregation_field=fields[0])

    # Step 7: Check for 'sorted by' or 'order by' (ORDER BY)
    order_by_field = None
    if token_index < len(tokens):
        if tokens[token_index] in ['sorted', 'order']:
            token_index += 1
            if token_index < len(tokens) and tokens[token_index] == 'by':
                token_index += 1
                if token_index < len(tokens):
                    order_by_field = tokens[token_index]
                    token_index += 1
                else:
                    return "Error: Expected field after 'by'."
            else:
                return "Error: Expected 'by' after 'sorted' or 'order'."

    ########### Construct the SQL query ###########
    sql_query_parts = []

    # SELECT clause
    select_clause = "SELECT "
    if is_aggregation:
        select_clause += f"{aggregation_function}({fields[0]}) AS {aggregation_function.lower()}"
    else:
        select_clause += ', '.join(fields)

    # FROM clause
    from_clause = "FROM "
    if len(tables) == 1:
        from_clause += tables[0]
    elif len(tables) == 2:
        from_clause += f"{tables[0]} JOIN {tables[1]} ON {join_condition}"
    else:
        return "Error: More than two tables are not supported."

    sql_query_parts.append(select_clause)
    sql_query_parts.append(from_clause)

    # WHERE clause 
    where_clause = None
    having_clause = None

    if condition:
        if is_aggregation and group_by_field:
            # Use HAVING clause
            having_clause = f"HAVING {condition}"
        else:
            # Use WHERE clause
            where_clause = f"WHERE {condition}"

    if where_clause:
        sql_query_parts.append(where_clause)

    # GROUP BY clause
    if group_by_field:
        sql_query_parts.append(f"GROUP BY {group_by_field}")

    # HAVING clause
    if having_clause:
        sql_query_parts.append(having_clause)

    # ORDER BY clause
    if order_by_field:
        if is_aggregation and order_by_field == aggregation_function.lower():
            # ORDER BY aggregation function
            order_by_clause = f"ORDER BY {aggregation_function}({fields[0]})"
        else:
            order_by_clause = f"ORDER BY {order_by_field}"
        sql_query_parts.append(order_by_clause)

    # LIMIT clause
    if limit_number:
        sql_query_parts.append(f"LIMIT {limit_number}")

    # Combine all parts
    sql_query = '\n'.join(sql_query_parts) + ';'

    return sql_query
def parse_sql_to_mongo(sql_query, database_info):
    sql_query = ' '.join(sql_query.strip().split())

    pipeline = []
    collection_name = ''
    aliases = {}  

    # Extract SELECT and FROM clauses
    select_match = re.search(r"SELECT\s+(.*?)\s+FROM\s+(\w+)(?:\s+AS\s+(\w+))?", sql_query, re.IGNORECASE)
    if select_match:
        select_clause = select_match.group(1)
        collection_name = select_match.group(2)
        main_alias = select_match.group(3) or collection_name
        aliases[main_alias] = collection_name
        print("main alias", aliases[main_alias])
    else:
        raise ValueError("Invalid SQL query: Cannot find SELECT and FROM clauses.")

    def clean_field(field):
        fields = field.split(".")
        if len(fields) > 2:
            field = fields[-2] + "." + fields[-1]
        if field.startswith(f"{main_alias}.") or field.startswith(f"{collection_name}."):
            return field.split('.')[-1]
        return field

    def clean_expression(expression):
        return expression.replace(f"{main_alias}.", "").replace(f"{collection_name}.", "")

    rest_of_query = sql_query[select_match.end():].strip()

    match_conditions = {}
    project_fields = {'_id': 0}  
    sort_fields = {}
    limit_value = None
    last_alias = main_alias  
    aggregation_fields = {}
    having_conditions = {}
    join_count = 0

    join_pattern = re.compile(
        r"(INNER|LEFT|RIGHT|FULL\s+OUTER|FULL)?\s*JOIN\s+(\w+)(?:\s+AS\s+(\w+)|\s+(\w+))?\s+ON\s+(.+?)(?=\s*(INNER|LEFT|RIGHT|JOIN|FULL|WHERE|GROUP BY|ORDER BY|LIMIT|;|$))",
        re.IGNORECASE | re.DOTALL
    )

    while True:
        join_match = join_pattern.search(rest_of_query)
        if not join_match:
            break

        join_type = (join_match.group(1) or '').strip().upper()
        table_name = join_match.group(2)
        alias = (join_match.group(3) or join_match.group(4) or table_name).strip()
        on_condition = join_match.group(5).strip()
        alias = alias or table_name
        aliases[alias] = table_name  

        left_field, right_field = [f.strip() for f in on_condition.split('=')]
        local_field = clean_field(left_field)
        foreign_field = right_field.split('.')[-1]
        if join_count == 0:
            local_field = local_field.split('.')[-1]

        lookup_stage = {
            '$lookup': {
                'from': table_name,
                'localField': local_field,
                'foreignField': foreign_field,
                'as': alias
            }
        }
        pipeline.append(lookup_stage)

        if join_type.upper() == 'INNER' or not join_type:
            pipeline.append({'$unwind': {
            "path": f"${alias}",
            "preserveNullAndEmptyArrays": True
        }})

        rest_of_query = rest_of_query[:join_match.start()] + rest_of_query[join_match.end():].strip()
        join_count += 1

    where_match = re.search(r"WHERE\s+(.*?)(\s+GROUP BY|\s+ORDER BY|\s+HAVING|\s*+LIMIT|;|$)", rest_of_query, re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1)
        conditions = [cond.strip() for cond in re.split(r' AND | and ', where_clause)]
        for condition in conditions:
            match_cond = re.match(r'(\w+\.?\w*)\s*(=|>|<|>=|<=|!=)\s*(.+)', condition)
            if match_cond:
                field, operator, value = match_cond.groups()
                field = clean_field(field) 
                operator_map = {'=': '$eq', '>': '$gt', '<': '$lt', '>=': '$gte', '<=': '$lte', '!=': '$ne'}
                mongo_operator = operator_map.get(operator)
                value = value.strip()
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                else:
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass  
                match_conditions[field] = {mongo_operator: value}
            else:
                raise ValueError(f"Unable to parse WHERE condition: {condition}")

    if match_conditions:
        pipeline.append({'$match': match_conditions})

    # Parse SELECT clause
    for select_field in select_clause.split(','):
        select_field = select_field.strip()
        agg_match = re.match(r"(SUM|AVG|COUNT|MIN|MAX)\((.*?)\)\s+AS\s+(\w+)", select_field, re.IGNORECASE)
        if agg_match:
            func, col, alias = agg_match.groups()
            col = clean_field(col)  
            func_map = {'SUM': '$sum', 'AVG': '$avg', 'COUNT': '$sum', 'MIN': '$min', 'MAX': '$max'}
            mongo_func = func_map[func.upper()]
            aggregation_fields[alias] = {mongo_func: f"${col}" if func.upper() != 'COUNT' else 1}
            project_fields[alias] = f"${alias}"
        else:
            clean_select_field = clean_field(select_field)
    
            # if the field is not from the main collection, add it to the project stage
            for alias, collection_name in aliases.items():
                if clean_select_field in database_info[selected_database][collection_name]["columns"]:
                    alias = collection_name
                    break
            # project_fields[clean_select_field] = f"${clean_select_field}"
            if alias == main_alias:
                project_fields[clean_select_field] = f"${clean_select_field}"
            elif "." in clean_select_field:
                field_name = clean_select_field.split('.')[-1]
                project_fields[field_name] = f"${clean_select_field}"
            else:
                project_fields[clean_select_field] = f"${alias}.{clean_select_field}"
            print(project_fields, select_field)

    # Extract GROUP BY clause
    group_by_match = re.search(r"GROUP BY\s+(.*?)(\s+HAVING|\s+ORDER BY|\s*+LIMIT|;|$)", rest_of_query, re.IGNORECASE)
    if group_by_match:
        group_by_fields = [clean_field(field.strip()) for field in group_by_match.group(1).split(',')]
        group_stage = {'$group': {'_id': {}}}
        for field in group_by_fields:
            if ' AS ' in field:
                field = field.split(' AS ')[1]
            group_stage['$group']['_id'][field] = f"${field}"
            project_fields[field] = f"$_id.{field}"

        for alias, agg_expr in aggregation_fields.items():
            group_stage['$group'][alias] = agg_expr

        pipeline.append(group_stage)

        # Extract HAVING clause
        having_match = re.search(r"HAVING\s+(.*?)(\s+ORDER BY|\s+LIMIT|;|$)", rest_of_query, re.IGNORECASE)
        if having_match:
            having_clause = having_match.group(1)
            having_conditions = []
            conditions = [cond.strip() for cond in re.split(r' AND | and ', having_clause)]
            for condition in conditions:
                match_cond = re.match(r'(\w+)\s*(=|>|<|>=|<=|!=)\s*(.+)', condition)
                if match_cond:
                    field, operator, value = match_cond.groups()
                    field = field.strip()
                    operator_map = {'=': '$eq', '>': '$gt', '<': '$lt', '>=': '$gte', '<=': '$lte', '!=': '$ne'}
                    mongo_operator = operator_map.get(operator)
                    value = value.strip()
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    else:
                        try:
                            value = float(value) if '.' in value else int(value)
                        except ValueError:
                            pass  
                    having_conditions.append({field: {mongo_operator: value}})
                else:
                    raise ValueError(f"Unable to parse HAVING condition: {condition}")
            print(having_conditions)

            if having_conditions:
                if len(having_conditions) > 1:
                    pipeline.append({'$match': {'$and': having_conditions}})
                else:
                    pipeline.append({'$match': having_conditions[0]})

        pipeline.append({'$project': project_fields})

    else:
        pipeline.append({'$project': project_fields})

    # Extract ORDER BY clause
    order_by_match = re.search(r"ORDER BY\s+(.*?)(ASC|DESC)?(\s+LIMIT|;|$)", rest_of_query, re.IGNORECASE)
    if order_by_match:
        order_by_fields = [clean_field(field.strip()) for field in order_by_match.group(1).split(',')]
        order_direction = 1
        direction = order_by_match.group(2)
        if direction and direction.upper() == 'DESC':
            order_direction = -1
        for field in order_by_fields:
            sort_fields[field] = order_direction

    if sort_fields:
        pipeline.append({'$sort': sort_fields})

    # Extract LIMIT clause
    limit_match = re.search(r"LIMIT\s+(\d+)", rest_of_query, re.IGNORECASE)
    if limit_match:
        limit_value = int(limit_match.group(1))
        pipeline.append({'$limit': limit_value})

    return collection_name, pipeline
def extract_column_names(sql_query):
    match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_query, re.IGNORECASE)
    if match:
        columns = match.group(1)  
        column_names = [col.strip() for col in re.split(r',\s*(?![^(]*\))', columns)]
        return column_names
    else:
        return []
def upload_data_to_db(df, filename, database_type, database_info, sample_size=5):
    df.columns = [col.lower() for col in df.columns]  # Standardize column names to lowercase
    column_types = {col: str(df[col].dtype) for col in df.columns}

    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'object': 'TEXT',
        'bool': 'BOOLEAN'
    }
    sqlite_columns = {col: type_mapping.get(str(dtype), 'TEXT') for col, dtype in column_types.items()}

    sample_data = df.head(sample_size).to_dict(orient='list')

    if database_type.lower() == 'sqlite' or database_type.lower() == 'mongodb':
        try:
            conn = sqlite3.connect('local_database.db')
            df.to_sql(filename, conn, if_exists='replace', index=False)
            if database_type.lower() == 'sqlite':
                st.success(f"Data from {filename} has been added to the SQLite database as table '{filename}'.")

            # Save the structure and sample data into the database_info dictionary
            database_info["local_database"][filename] = {
                "columns": sqlite_columns,
                "samples": sample_data
            }

            conn.commit()
        except Exception as e:
            st.error(f"An error occurred while uploading to SQLite: {e}")
        finally:
            conn.close()

        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['local_database']  # Specify the database name

            # Upload data to MongoDB collection
            data = df.to_dict(orient='records')
            db[filename].insert_many(data)
            if database_type.lower() == 'mongodb':
                st.success(f"Data from {filename} has been uploaded to MongoDB collection '{filename}'.")

            # Save the structure and sample data into the database_info dictionary
            database_info["local_database"][filename] = {
                "columns": sqlite_columns,
                "samples": sample_data
            }
        except Exception as e:
            st.error(f"An error occurred while uploading to MongoDB: {e}")
        finally:
            client.close()  # Ensure the MongoDB client is closed

    else:
        st.error("Unsupported database type. Please use 'SQLite' or 'MongoDB'.")

    # Display the database_info structure
    st.write("Database structure saved:", database_info)

suggestions = [
    "Find studentid, firstName, lastName of students with studentID equal to 1", # Where clause
    "Find 10 firstName, lastName of students", # Limit clause
    "Calculate sum of credithours of courses by instructorname", # Group By clause
    "Find firstName, lastName, courseid of students, enrollments", # Join clause
    "Show 5 firstname, lastname of students, enrollments sorted by lastName", # Order By clause
    '\\explore',
    '\\sample',
    '\\groupby',
    '\\join',
    '\\where',
    '\\orderby',
    '\\limit',
    '\\having'
]
##################################################################################################################################################################
# Streamlit App Interface
##################################################################################################################################################################

st.title("Database Chat Query App")
# st.markdown("This app allows you to upload a CSV file, interact with a database, and retrieve information using natural language queries.")
# Select Database Type
database_type = st.radio("Select Database Type:", ("MongoDB", "SQLite"))

show_upload = st.checkbox("Show Upload Section", value=True)

# Upload CSV File
if show_upload:
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("CSV Preview:", df.head())

        filename = uploaded_file.name.split(".")[0]  
        if st.button("Upload Data to Database"):
            upload_data_to_db(df, filename, database_type, database_info)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.subheader("Chat with the Database")

user_input = st.text_input("You:", key="input")

if user_input:
    # keep only the latest chat history
    st.session_state.chat_history = [{"role": "user", "content": user_input}]

    sql_queries = process_command(user_input, database_info)
    
    for sql_query in sql_queries:
        if database_type == "SQLite":
            response_data = get_sqlite_data(selected_database, sql_query)
            column_name = extract_column_names(sql_query)
            response_data = pd.DataFrame(response_data, columns=column_name)
        elif database_type == "MongoDB":
            response_data = get_mongo_data(selected_database, sql_query)
        
        if not isinstance(response_data, pd.DataFrame):
            df = pd.DataFrame([response_data])
        else:
            df = response_data

        st.session_state.chat_history.append({"role": "bot", "content": {"query": sql_query, "data": df}})

# Display chat history
for entry in st.session_state.chat_history:
    role, content = entry["role"], entry["content"]
    if role == "user":
        pass
    else:
        if isinstance(content, dict):  
            st.write("Bot:")
            st.code(f"{content['query']}", language="sql")  
            st.dataframe(content["data"].head()) 
        else:
            st.write(f"Bot: {content}")