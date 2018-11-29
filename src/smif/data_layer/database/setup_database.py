"""Set up a smif database
"""
import subprocess
from os import environ, listdir
from os.path import dirname, isfile, join, realpath


def run_migrations(migrations_dir, migration_direction, database_name):
    """Run specified database migrations
    """

    # run migrations given parameter passed
    if migration_direction == 'up':
        ran = up_migrations(migrations_dir, database_name)
    elif migration_direction == 'down':
        ran = down_migrations(migrations_dir, database_name)

    return ran


def down_migrations(working_path, database_name):
    """Run down migrations to resolve database to blank
    """

    # run key file to remove foreign keys from database
    file = 'remove-foreign_keys.sql'
    # check file exists
    if isfile(join(working_path, file)):
        # run sql file silently
        subprocess.run(['psql', '-d', database_name, '-q', '-f', join(working_path, file)])
    else:
        return False

    # loop through files to remove tables
    for file in listdir(working_path):

        # check if file is a down migration
        if file[0:4] == 'down':
            # run sql file silently
            subprocess.run(['psql', '-d', database_name, '-q', '-f', join(working_path, file)])

    return True


def up_migrations(working_path, database_name):
    """Run up migrations to build database
    """

    # declare name of foreign key file
    # run key file to add foreign keys between relations
    fky_file = 'add-foreign_keys.sql'

    # check file exists - if not stop now
    if isfile(join(working_path, fky_file)) is False:
        return False

    # loop through files
    for file in listdir(working_path):

        # check if file is a down migration
        if file[0:2] == 'up':
            # run sql file silently
            subprocess.run(['psql', '-d', database_name, '-q', '-f', join(working_path, file)])

    # run key file to add foreign keys between relations
    # run sql file silently
    subprocess.run(['psql', '-d', database_name, '-q', '-f', join(working_path, fky_file)])

    return True


def main(database_name):
    """Run database setup

    Relies on standard Postgres environment variables being set to configure the database
    connection: https://www.postgresql.org/docs/current/libpq-envars.html
    """
    # get the base directory and build the path for the database config files
    migrations_dir = join(dirname(realpath(__file__)), 'migrations')

    # run migrations - delete existing relations than add new set
    # run the migrations - down first in case anything exists already
    # run migrations - down
    run_migrations(migrations_dir, 'down', database_name)

    # run the migrations - up to build the database
    # run migrations - up
    run_migrations(migrations_dir, 'up', database_name)

    # confirm migrations have been run
    print('Finished building database')


if __name__ == '__main__':
    # run main when file called
    try:
        DB_NAME = environ['PGDATABASE']
    except KeyError:
        print('PGDATABASE environment variable must be set to configure database name')
        exit(1)
    main(DB_NAME)
