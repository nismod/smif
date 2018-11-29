"""Set up a smif database
"""
import subprocess
from os import listdir
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
        subprocess.run(['psql', '-U', 'vagrant', '-d', '%s' %
                        database_name, '-q', '-f', join(working_path, file)])
    else:
        return False

    # loop through files to remove tables
    for file in listdir(working_path):

        # check if file is a down migration
        if file[0:4] == 'down':
            # run sql file silently
            subprocess.run(['psql', '-U', 'vagrant', '-d', '%s' %
                            database_name, '-q', '-f', join(working_path, file)])

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
            subprocess.run(['psql', '-U', 'vagrant', '-d', '%s' %
                            database_name, '-q', '-f', join(working_path, file)])

    # run key file to add foreign keys between relations
    # run sql file silently
    subprocess.run(['psql', '-U', 'vagrant', '-d', '%s' %
                    database_name, '-q', '-f', join(working_path, fky_file)])

    return True


def main():
    """Run database setup process
    """

    # get the base directory and build the path for the database config files
    config_dir = dirname(realpath(__file__))

    # database name
    db_name = 'nismod_smif_config'

    # build the database if it does not exist
    cmd = """psql -U vagrant -tc \
        "SELECT 1 FROM pg_database WHERE datname = '%s';" \
        | grep -q 1 || psql -U vagrant -c "CREATE DATABASE %s;"
        """ % (db_name, db_name)
    subprocess.run(cmd, shell=True)

    # run migrations - delete existing relations than add new set
    # run the migrations - down first in case anything exists already
    # run migrations - down
    run_migrations(config_dir + '/migrations', 'down', db_name)

    # run the migrations - up to build the database
    # run migrations - up
    run_migrations(config_dir + '/migrations', 'up', db_name)

    # confirm migrations have been run
    subprocess.run(['echo', 'Finished building database'])


if __name__ == '__main__':
    # run main when file called
    main()
