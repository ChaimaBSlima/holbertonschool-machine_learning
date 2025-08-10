-- Write a SQL script that creates a stored procedure
-- AddBonus that adds a new correction for a student.
-- Requirements:
-- Procedure AddBonus is taking 3 inputs (in this order):
-- user_id, a users.id value (you can assume user_id is linked to an existing users)
-- project_name, a new or already exists projects - if no projects.name found in the table, you should create it
-- score, the score value for the correction
-- Context: Write code in SQL is a nice level up!

delimiter //
CREATE PROCEDURE AddBonus(IN user_id INT, project_name VARCHAR(255), score INT)
BEGIN
    SET @count = (SELECT COUNT(*) FROM projects WHERE projects.name LIKE project_name);
    IF @count = 0 THEN
        INSERT INTO projects(name)
        VALUES (project_name);
    END IF;
    INSERT INTO corrections
    VALUES (user_id, (SELECT id FROM projects WHERE name = project_name), score);
END
//
delimiter ;
