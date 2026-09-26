use std::{
    borrow::Cow,
    ffi::OsString,
    io::{self, BufWriter, Write},
    mem::forget,
    path::{self, Path, PathBuf},
};

#[cfg(windows)]
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use fs_err::File;
use memofs::Vfs;
use rayon::prelude::*;
use rbx_dom_weak::{types::Ref, Ustr};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

use crate::{
    serve_session::ServeSession,
    snapshot::{AppliedPatchSet, InstanceWithMeta, RojoTree},
};

use super::resolve_path;

const PATH_STRIP_FAILED_ERR: &str = "Failed to create relative paths for project file!";
const ABSOLUTE_PATH_FAILED_ERR: &str = "Failed to turn relative path into absolute path!";

/// Representation of a node in the generated sourcemap tree.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourcemapNode<'a> {
    name: &'a str,
    class_name: Ustr,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "crate::path_serializer::serialize_vec_absolute"
    )]
    file_paths: Vec<Cow<'a, Path>>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    children: Vec<SourcemapNode<'a>>,
}

/// Generates a sourcemap file from the Rojo project.
#[derive(Debug, Parser)]
pub struct SourcemapCommand {
    /// Path to the project to use for the sourcemap. Defaults to the current
    /// directory.
    #[clap(default_value = "")]
    pub project: PathBuf,

    /// Where to output the sourcemap. Omit this to use stdout instead of
    /// writing to a file.
    ///
    /// Should end in .json.
    #[clap(long, short)]
    pub output: Option<PathBuf>,

    /// If non-script files should be included or not. Defaults to false.
    #[clap(long)]
    pub include_non_scripts: bool,

    /// Whether to automatically recreate a snapshot when any input files change.
    #[clap(long)]
    pub watch: bool,

    /// Whether the sourcemap should use absolute paths instead of relative paths.
    #[clap(long)]
    pub absolute: bool,
}

impl SourcemapCommand {
    pub fn run(self) -> anyhow::Result<()> {
        let project_path = fs_err::canonicalize(resolve_path(&self.project)?)?;

        log::trace!("Constructing filesystem with StdBackend");
        let vfs = Vfs::new_default()?;
        vfs.set_watch_enabled(self.watch);

        log::trace!("Setting up session for sourcemap generation");
        let session = ServeSession::new(vfs, project_path)?;
        let mut cursor = session.message_queue().cursor();

        let filter = if self.include_non_scripts {
            filter_nothing
        } else {
            filter_non_scripts
        };

        // Pre-build a rayon threadpool with a low number of threads to avoid
        // dynamic creation overhead on systems with a high number of cpus.
        log::trace!("Setting rayon global threadpool");
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().min(6))
            .build_global()
            .ok();

        log::trace!("Writing initial sourcemap");
        write_sourcemap(&session, self.output.as_deref(), filter, self.absolute)?;

        if self.watch {
            log::trace!("Setting up runtime for watch mode");
            let rt = Runtime::new().context("Failed to start the async runtime for watch mode")?;

            loop {
                let receiver = session.message_queue().subscribe(cursor);
                let (new_cursor, patch_set) = match rt.block_on(receiver) {
                    Ok(message) => message,
                    // The message queue was dropped, so there is nothing left
                    // to watch. Stop watching gracefully.
                    Err(_) => break,
                };
                cursor = new_cursor;

                if patch_set_affects_sourcemap(&session, &patch_set, filter) {
                    write_sourcemap(&session, self.output.as_deref(), filter, self.absolute)?;
                }
            }
        }

        // Avoid dropping ServeSession: it's potentially VERY expensive to drop
        // and we're about to exit anyways.
        forget(session);

        Ok(())
    }
}

fn filter_nothing(_instance: &InstanceWithMeta) -> bool {
    true
}

fn filter_non_scripts(instance: &InstanceWithMeta) -> bool {
    matches!(
        instance.class_name().as_str(),
        "Script" | "LocalScript" | "ModuleScript"
    )
}

fn patch_set_affects_sourcemap(
    session: &ServeSession,
    patch_set: &[AppliedPatchSet],
    filter: fn(&InstanceWithMeta) -> bool,
) -> bool {
    let tree = session.tree();

    // The tree may already include patches applied after these ones, so an
    // instance they added or updated can be gone again. We can't check the
    // filter for it, just like for removed instances.
    let passes_filter = |id: Ref| {
        tree.get_instance(id)
            .is_none_or(|instance| filter(&instance))
    };

    // A sourcemap has probably changed when:
    patch_set.par_iter().any(|set| {
        // 1. An instance was removed, in which case it will no
        // longer exist in the tree and we cant check the filter
        !set.removed.is_empty()
            // 2. A newly added instance passes the filter
            || set.added.iter().any(|&referent| passes_filter(referent))
            // 3. An existing instance has its class name, name,
            // or file paths changed, and passes the filter
            || set.updated.iter().any(|updated| {
                let changed = updated.changed_class_name.is_some()
                    || updated.changed_name.is_some()
                    || updated.changed_metadata.is_some();
                changed && passes_filter(updated.id)
            })
    })
}

fn recurse_create_node<'a>(
    tree: &'a RojoTree,
    referent: Ref,
    project_dir: &Path,
    filter: fn(&InstanceWithMeta) -> bool,
    use_absolute_paths: bool,
) -> Option<SourcemapNode<'a>> {
    let instance = tree.get_instance(referent).expect("instance did not exist");

    let children: Vec<_> = instance
        .children()
        .par_iter()
        .filter_map(|&child_id| {
            recurse_create_node(tree, child_id, project_dir, filter, use_absolute_paths)
        })
        .collect();

    // If this object has no children and doesn't pass the filter, it doesn't
    // contain any information we're looking for.
    if children.is_empty() && !filter(&instance) {
        return None;
    }

    let file_paths = instance
        .metadata()
        .relevant_paths
        .iter()
        // Not all paths listed as relevant are guaranteed to exist.
        .filter(|path| path.is_file())
        .map(|path| path.as_path());

    let mut output_file_paths: Vec<Cow<'a, Path>> =
        Vec::with_capacity(instance.metadata().relevant_paths.len());

    if use_absolute_paths {
        // It's somewhat important to note here that `path::absolute` takes in a Path and returns a PathBuf
        for val in file_paths {
            output_file_paths.push(Cow::Owned(
                path::absolute(val).expect(ABSOLUTE_PATH_FAILED_ERR),
            ));
        }
    } else {
        for val in file_paths {
            output_file_paths.push(Cow::from(
                pathdiff::diff_paths(val, project_dir).expect(PATH_STRIP_FAILED_ERR),
            ));
        }
    };

    Some(SourcemapNode {
        name: instance.name(),
        class_name: instance.class_name(),
        file_paths: output_file_paths,
        children,
    })
}

fn write_sourcemap(
    session: &ServeSession,
    output: Option<&Path>,
    filter: fn(&InstanceWithMeta) -> bool,
    use_absolute_paths: bool,
) -> anyhow::Result<()> {
    let tree = session.tree();

    let root_node = recurse_create_node(
        &tree,
        tree.get_root_id(),
        session.root_dir(),
        filter,
        use_absolute_paths,
    );

    if let Some(output_path) = output {
        // Write to a temporary file next to the output and rename it over the
        // output. The rename is atomic, so tools reading the sourcemap while
        // it's rewritten see either the old one or the new one, never a
        // truncated file. The temporary file's name is fixed so that one left
        // behind by a killed process is overwritten by the next write.
        let mut temp_name = OsString::from(".");
        temp_name.push(
            output_path
                .file_name()
                .context("The sourcemap output path has no file name")?,
        );
        temp_name.push(".tmp");
        let temp_path = output_path.with_file_name(temp_name);

        {
            let mut file = BufWriter::new(File::create(&temp_path)?);
            serde_json::to_writer(&mut file, &root_node)?;
            file.flush()?;
        }
        replace_file(&temp_path, output_path).with_context(|| {
            format!(
                "Failed to replace {} with {}",
                output_path.display(),
                temp_path.display()
            )
        })?;

        println!("Created sourcemap at {}", output_path.display());
    } else {
        let output = serde_json::to_string(&root_node)?;
        println!("{}", output);
    }

    Ok(())
}

/// Renames `from` over `to`, replacing it.
///
/// On Windows, a file can't be replaced while another process has it open
/// without sharing delete access, which is how many programs read files. They
/// only hold it for as long as a read takes, so retry for a moment.
fn replace_file(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        const ERROR_ACCESS_DENIED: i32 = 5;
        const ERROR_SHARING_VIOLATION: i32 = 32;
        const RETRY_FOR: Duration = Duration::from_secs(2);

        let start = Instant::now();
        loop {
            match std::fs::rename(from, to) {
                Err(err)
                    if matches!(
                        err.raw_os_error(),
                        Some(ERROR_ACCESS_DENIED | ERROR_SHARING_VIOLATION)
                    ) && start.elapsed() < RETRY_FOR =>
                {
                    std::thread::sleep(Duration::from_millis(10));
                }
                result => return result,
            }
        }
    }

    #[cfg(not(windows))]
    std::fs::rename(from, to)
}

#[cfg(test)]
mod test {
    use crate::cli::sourcemap::{filter_non_scripts, patch_set_affects_sourcemap, SourcemapNode};
    use crate::cli::SourcemapCommand;
    use crate::serve_session::ServeSession;
    use crate::snapshot::{AppliedPatchSet, AppliedPatchUpdate};
    use insta::internals::Content;
    use memofs::Vfs;
    use rbx_dom_weak::types::Ref;
    use std::io::Read;
    use std::path::Path;

    #[cfg(windows)]
    #[test]
    fn replace_file_waits_for_readers_that_block_replacing() {
        use std::{os::windows::fs::OpenOptionsExt, thread, time::Duration};

        // Programs reading through the C runtime, like luau-lsp, open files
        // without FILE_SHARE_DELETE, so the file can't be replaced meanwhile.
        const FILE_SHARE_READ: u32 = 0x1;
        const FILE_SHARE_WRITE: u32 = 0x2;

        let sourcemap_dir = tempfile::tempdir().unwrap();
        let temp_path = sourcemap_dir.path().join(".sourcemap.json.tmp");
        let sourcemap_output = sourcemap_dir.path().join("sourcemap.json");
        fs_err::write(&temp_path, "new sourcemap").unwrap();
        fs_err::write(&sourcemap_output, "old sourcemap").unwrap();

        let reader = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .open(&sourcemap_output)
            .unwrap();
        let reader = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            drop(reader);
        });

        super::replace_file(&temp_path, &sourcemap_output).unwrap();
        reader.join().unwrap();

        assert_eq!(
            fs_err::read_to_string(&sourcemap_output).unwrap(),
            "new sourcemap"
        );
    }

    #[test]
    fn patches_for_instances_that_are_gone_affect_the_sourcemap() {
        let project_path = fs_err::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test-projects")
                .join("relative_paths")
                .join("project"),
        )
        .unwrap();
        let vfs = Vfs::new_default().unwrap();
        vfs.set_watch_enabled(false);
        let session = ServeSession::new(vfs, project_path).unwrap();

        // Added or updated by a patch, then removed by a later one that the
        // tree already includes.
        let gone = Ref::new();

        let mut added = AppliedPatchSet::new();
        added.added.push(gone);
        assert!(patch_set_affects_sourcemap(
            &session,
            &[added],
            filter_non_scripts
        ));

        let mut update = AppliedPatchUpdate::new(gone);
        update.changed_name = Some("Gone".to_owned());
        let mut updated = AppliedPatchSet::new();
        updated.updated.push(update);
        assert!(patch_set_affects_sourcemap(
            &session,
            &[updated],
            filter_non_scripts
        ));
    }

    #[test]
    fn replaces_the_output_instead_of_rewriting_it() {
        let sourcemap_dir = tempfile::tempdir().unwrap();
        let sourcemap_output = sourcemap_dir.path().join("sourcemap.json");
        fs_err::write(&sourcemap_output, "old sourcemap").unwrap();
        // Left behind by a process that was killed while writing.
        fs_err::write(sourcemap_dir.path().join(".sourcemap.json.tmp"), "partial").unwrap();

        // A tool that opened the old sourcemap just before it was rewritten.
        let mut reader = fs_err::File::open(&sourcemap_output).unwrap();

        let project_path = fs_err::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test-projects")
                .join("relative_paths")
                .join("project"),
        )
        .unwrap();
        let sourcemap_command = SourcemapCommand {
            project: project_path,
            output: Some(sourcemap_output.clone()),
            include_non_scripts: false,
            watch: false,
            absolute: false,
        };
        assert!(sourcemap_command.run().is_ok());

        // Rewriting the file in place would truncate it under the reader.
        let mut old_contents = String::new();
        reader.read_to_string(&mut old_contents).unwrap();
        assert_eq!(old_contents, "old sourcemap");

        let raw_sourcemap_contents = fs_err::read_to_string(&sourcemap_output).unwrap();
        serde_json::from_str::<SourcemapNode>(&raw_sourcemap_contents).unwrap();

        // Both the stale temporary file and the new one were renamed into place.
        let file_names: Vec<_> = fs_err::read_dir(sourcemap_dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(file_names, ["sourcemap.json"]);
    }

    #[test]
    fn maps_relative_paths() {
        let sourcemap_dir = tempfile::tempdir().unwrap();
        let sourcemap_output = sourcemap_dir.path().join("sourcemap.json");
        let project_path = fs_err::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test-projects")
                .join("relative_paths")
                .join("project"),
        )
        .unwrap();
        let sourcemap_command = SourcemapCommand {
            project: project_path,
            output: Some(sourcemap_output.clone()),
            include_non_scripts: false,
            watch: false,
            absolute: false,
        };
        assert!(sourcemap_command.run().is_ok());

        let raw_sourcemap_contents = fs_err::read_to_string(sourcemap_output.as_path()).unwrap();
        let sourcemap_contents =
            serde_json::from_str::<SourcemapNode>(&raw_sourcemap_contents).unwrap();
        insta::assert_json_snapshot!(sourcemap_contents);
    }

    #[test]
    fn maps_absolute_paths() {
        let sourcemap_dir = tempfile::tempdir().unwrap();
        let sourcemap_output = sourcemap_dir.path().join("sourcemap.json");
        let project_path = fs_err::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test-projects")
                .join("relative_paths")
                .join("project"),
        )
        .unwrap();
        let sourcemap_command = SourcemapCommand {
            project: project_path,
            output: Some(sourcemap_output.clone()),
            include_non_scripts: false,
            watch: false,
            absolute: true,
        };
        assert!(sourcemap_command.run().is_ok());

        let raw_sourcemap_contents = fs_err::read_to_string(sourcemap_output.as_path()).unwrap();
        let sourcemap_contents =
            serde_json::from_str::<SourcemapNode>(&raw_sourcemap_contents).unwrap();
        insta::assert_json_snapshot!(sourcemap_contents, {
            ".**.filePaths" => insta::dynamic_redaction(|mut value, _path| {
                let mut paths_count = 0;

                match value {
                    Content::Seq(ref mut vec) => {
                        for path in vec.iter().map(|i| i.as_str().unwrap()) {
                            assert_eq!(fs_err::canonicalize(path).is_ok(), true, "path was not valid");
                            assert_eq!(Path::new(path).is_absolute(), true, "path was not absolute");

                            paths_count += 1;
                        }
                    }
                    _ => panic!("Expected filePaths to be a sequence"),
                }
                format!("[...{} path{} omitted...]", paths_count, if paths_count != 1 { "s" } else { "" } )
            })
        });
    }
}
